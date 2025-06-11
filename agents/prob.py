import math, random
from statistics import NormalDist
from game.engine.hand_evaluator import HandEvaluator
from game.engine.card import Card
from game.engine.table import Table
from game.engine.pay_info import PayInfo
from game.engine.player import Player

from typing import Optional, Any, List, Dict, Tuple

MU_ROUND   = 4.5
SIGMA_ROUND = 20.0

# ref: https://www.csie.ntu.edu.tw/~b09902097/HoldemAgent.pdf

def game_rate(t_chip, remain_round) -> float:
    if remain_round == 0:
        return 1.0 if t_chip > 0 else 0.0
    mu = remain_round * MU_ROUND
    sigma = math.sqrt(remain_round) * SIGMA_ROUND
    return 1.0 - NormalDist(mu, sigma).cdf(-t_chip)

def round_rate(my_uuid: str, table: Table) -> float:
    community: list[Card] = table.get_community_card()
    deck_list: list[Card] = table.deck.deck[:]
    random.shuffle(deck_list)

    remaining_players: List[Player] = [p for p in table.seats.players if p.pay_info.status != PayInfo.FOLDED]

    # deal unknown hole cards
    for player in remaining_players:
        if len(player.hole_card)==0:
            player.hole_card = [deck_list.pop(), deck_list.pop()]

    # finish board to 5 cards
    board: list[Card] = community[:] # copy list
    while len(board) < 5:
        board.append(deck_list.pop())

    scores = {p.uuid:HandEvaluator.eval_hand(p.hole_card, board) for p in remaining_players}
    my_score = scores[my_uuid]
    scores_list = list(scores.values())
    best = max(scores_list)
    n_best = scores_list.count(best)
    if my_score == best:
        win_share = 1.0 / n_best
    else:
        win_share = 0.0
    return win_share



class ProbAgent:
    @staticmethod
    def act(sim_state: Dict[str, Any], player_index: int, actions: List[Dict[str, Any]], round_start_stacks: Dict[str, int]) -> Tuple[str, int]:
        call_amt = actions[1]["amount"]
        raise_rng = actions[2]["amount"]
        min_r, max_r = raise_rng["min"], raise_rng["max"]

        player: Player = sim_state["table"].seats.players[player_index]

        initial_stack  = 1000
        round_start_stack = round_start_stacks[player.uuid]
        t_base = round_start_stack - initial_stack # money earned at the start of the round
        bet = player.pay_info.amount # money the player has bet in the current round

        # round-rate p
        p = round_rate(player.uuid, sim_state["table"])
        # rounds left
        C_left = 20 - sim_state["round_count"]   # max_round=20

        pot: int = sum(player.pay_info.amount for player in sim_state["table"].seats.players)

        # note: ev == Expected Value

        # fold
        t_fold = t_base - bet
        ev_fold: float = game_rate(t_fold, C_left - 1)

        # call
        add_amount = call_amt - player.paid_sum() # amount of chips to pay if called
        t_plus  = t_base - bet + pot
        t_minus = t_base - bet - add_amount
        ev_call: float = p * game_rate(t_plus, C_left - 1) + (1-p) * game_rate(t_minus, C_left - 1)

        # raise
        remaining_players_except_me = [(p, p.paid_sum()) for i,p in enumerate(sim_state["table"].seats.players) if i!=player_index and p.pay_info.status != PayInfo.FOLDED]
        def ev_raise(raise_amount: int) -> float:
            add_amount = raise_amount - player.paid_sum()
            opponent_add_amount = sum(raise_amount - paid_sum for pl,paid_sum in remaining_players_except_me if pl.stack >= raise_amount-paid_sum) # assuming all successors call
            t_plus  = t_base - bet + pot + opponent_add_amount # this should be our opponent's add amount
            t_minus = t_base - bet - add_amount
            return p * game_rate(t_plus, C_left - 1) + (1-p) * game_rate(t_minus, C_left - 1)

        # pick best EV
        best_ev, best_action, best_amt = ev_fold, "fold", 0
        if ev_call > best_ev:
            best_ev, best_action, best_amt = ev_call, "call", call_amt
        if min_r != -1:
            raise_candidates = [min_r, int(pot/2), pot, max_r]
            for x in raise_candidates:
                if not (min_r <= x <= max_r):
                    continue
                ev_r = ev_raise(x)
                if ev_r > best_ev:
                    best_ev, best_action, best_amt = ev_r, "raise", x

        return best_action, best_amt
