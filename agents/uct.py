import copy, math, random, time
from game.engine.round_manager import RoundManager
from game.engine.card import Card
from game.engine.deck import Deck
from game.engine.table import Table
from game.engine.player import Player
from game.engine.pay_info import PayInfo
from game.engine.hand_evaluator import HandEvaluator
from game.engine.action_checker import ActionChecker
from game.engine.poker_constants import PokerConstants as Const
from typing import Optional, Any, List, Dict, Tuple, Set, Union

from agents.prob import ProbAgent

# -- for UCB calculation --
EXPLORATION_COEFF = math.sqrt(2)
K = 0.4
C_CAP = 99999


MAX_RAISES_PER_NODE = 10
LOSE_PENALTY = 2
RAISE_LIMITER = 2/3


class UCTState:
    def __init__(self, engine_state: Dict[str, Any], my_index: int, round_start_stacks: Dict[str, int], parent: Optional['UCTState']=None, move: Optional[Tuple[str, int]]=None, root_stacks: Optional[List[int]] = None):
        self.state: Dict[str, Any] = engine_state # must be deep-copied already!!!!!!!!!!!
        self.my_index: int = my_index

        # immutable baseline stacks (for payoff) captured at ROOT
        self.root_stacks: List[int] = root_stacks if root_stacks is not None else \
                                        [p.stack for p in self.state["table"].seats.players]

        self.round_start_stacks: Dict[str, int] = round_start_stacks

        # Tree Related things:
        self.parent: Optional['UCTState'] = parent
        self.move: Optional[Tuple[str, int]] = move # (action, amount) that led here
        self.children: List['UCTState'] = []
        self.visits: int = 0
        self.wins: float = 0.0
        self.untried: List[Tuple[str, int]] = self.enumerate_moves()
        self.pot_size: int = sum(p.pay_info.amount for p in self.state["table"].seats.players)
        self.remaining_betting_potential: int = sum(p.stack for p in self.state["table"].seats.players if p.pay_info.status != PayInfo.FOLDED)


    def ucb(self) -> float:
        if self.visits == 0:
            return float("inf")
        return self.wins / self.visits + EXPLORATION_COEFF * math.sqrt(math.log(self.parent.visits) / self.visits)

    def ucb_smooth(self) -> float:
        if self.visits == 0:
            return float("inf")
        parent_mean = self.parent.wins / self.parent.visits
        eta = 100 / (100 + self.visits)
        mixed = (1-eta) * (self.wins / self.visits) + eta * parent_mean
        explore = EXPLORATION_COEFF * math.sqrt(math.log(self.parent.visits) / self.visits)
        return mixed + explore

    def ucb_dynamic_c(self, k=K, C_cap=C_CAP) -> float:
        if self.visits == 0:
            return float("inf")
        c = min(C_cap, self.pot_size + k * self.remaining_betting_potential)
        #print(f"c == {c}")
        exploit = self.wins / self.visits
        explore = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore

    def enumerate_moves(self) -> List[Tuple[str, int]]:
        if self.is_terminal():
            return list()
        actions: List[Dict[str, Any]] = ActionChecker.legal_actions(
                    self.state["table"].seats.players,
                    self.state["next_player"],
                    self.state["small_blind_amount"]
        )
        moves: List[Tuple[str, int]] = []
        for act in actions:
            if act["action"] in ("fold", "call"):
                moves.append((act["action"], act["amount"]))
            else: # action == raise
                rng = act["amount"]
                if rng["min"] == -1: # raising not allowed
                    continue

                min_r = rng["min"]
                max_r = round(min_r + (rng["max"]-min_r) * RAISE_LIMITER)
                span: int = max_r - min_r

                # generate every possible amount if the span is small enough
                if span <= MAX_RAISES_PER_NODE:
                    moves.extend(('raise', i) for i in range(min_r, max_r + 1))
                else:
                    step: float = span/(MAX_RAISES_PER_NODE-1)
                    moves.extend(('raise', round(min_r + step*i)) for i in range(0, MAX_RAISES_PER_NODE))

        random.shuffle(moves) # ensures stochastic expansion
        return moves

    def expand(self) -> 'UCTState':
        action, amount = self.untried.pop()
        child_state, _ = RoundManager.apply_action(
            original_state=self.state,
            action=action,
            bet_amount=amount
        )
        child = UCTState(engine_state=child_state, my_index=self.my_index, round_start_stacks=self.round_start_stacks, parent=self, move=(action, amount), root_stacks=self.root_stacks)
        self.children.append(child)
        return child

    def is_terminal(self) -> bool:
        return self.state["street"] == Const.Street.FINISHED

    def rollout(self) -> Union[int,float]:
        """
        simulation
        step1: randomise all unseen cards
        step2:run every player *check/call* to showdown
        step3: score with HandEvaluator
        """
        #print("rollout")
        #sim_state = copy.deepcopy(self.state) # TODO: prune this. This operation is costly
        sim_state = copy.copy(self.state)
        sim_state['table'] = copy.copy(self.state['table'])
        sim_state['table'].deck =  Deck.deserialize(self.state['table'].deck.serialize())
        #print('copy done')

        # deal unknown cards & board
        random.shuffle(sim_state['table'].deck.deck)

        for idx, player in enumerate(sim_state["table"].seats.players):
            if idx != self.my_index:
                player.hole_card = sim_state['table'].deck.draw_cards(2)

        assert self.is_terminal() or len(sim_state['table'].seats.players[self.my_index].hole_card)==2
        #for _ in range(5-len(sim_state["table"].get_community_card())):
        #    sim_state["table"].add_community_card(sim_state['table'].deck.draw_card())

        # every player acts
        while sim_state["street"] != Const.Street.FINISHED:
            seat_idx: int = sim_state["next_player"]
            actions: List[Dict[str, Any]] = ActionChecker.legal_actions(
                sim_state["table"].seats.players,
                sim_state["next_player"],
                sim_state["small_blind_amount"]
            )
            #call_action = actions[1]
            action, amount = ProbAgent.act(sim_state, seat_idx, actions, self.round_start_stacks)
            sim_state, _ = RoundManager.apply_action(
                original_state=sim_state,
                action=action,
                bet_amount=amount
            )

        # calculate the money earned/lost
        stack_after: int = sim_state["table"].seats.players[self.my_index].stack
        reward = (stack_after - self.root_stacks[self.my_index])

        if stack_after < self.root_stacks[self.my_index]:
            reward *= LOSE_PENALTY

        return reward


class UCT:
    @staticmethod
    def search(root_state: Dict[str, Any], my_index: int, round_start_stacks: Dict[str, int], time_limit: Union[float,int]=9.9, verbose=False) -> Tuple[str, int]:
        root: UCTState = UCTState(engine_state=root_state, my_index=my_index, round_start_stacks=round_start_stacks)
        end: float = time.time() + time_limit
        iteration = 0

        while time.time() < end:
            if verbose:
                iteration += 1
                print(f"\n=== UCT iteration {iteration} ===")
                for child in root.children:
                    wr = child.wins / child.visits
                    act, amt = child.move
                    print(f"{act:5} {amt:5}   UCB={child.ucb_dynamic_c():8.3f}   win_rate={wr:10.3f}   visits={child.visits}")
                # also show yet-unexpanded moves
                for act, amt in root.untried:
                    print(f"{act:5} {amt:5}   UCB=  ------   win_rate=  --------   visits=0  (untried)")

            node: UCTState = root
            # UCT: SELECTION
            while not node.is_terminal() and not node.untried: # nonterminal and fully expanded
                #print(f"fully expanded with #children={len(node.children)}")
                node = max(node.children, key=lambda n: n.ucb_dynamic_c()) # then find best child

            # UCT: EXPANSION
            if node.untried and not node.is_terminal(): # if not fully expanded and is not terminal
                node = node.expand() # expand it
                #print("expanded")

            # UCT: SIMULATION
            reward = node.rollout()
            #print("reward:", reward)

            # UCT: BACKWARD-PROPOGATION
            while node is not None:
                node.visits += 1
                node.wins += reward
                node = node.parent


        # pick best child by average value (no exploration here)
        best = max(root.children, key=lambda n: n.wins / n.visits)
        return best.move

