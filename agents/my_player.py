from game.players import BasePokerPlayer
from game.engine.round_manager import RoundManager
from game.engine.action_checker import ActionChecker
from game.engine.table import Table
from game.engine.player import Player
from game.engine.pay_info import PayInfo
from game.engine.seats import Seats
from game.engine.card import Card
from game.engine.deck import Deck
from game.engine.poker_constants import PokerConstants as Const

from agents.uct import UCTState, UCT
from agents.prob import ProbAgent
from agents.config import Tunables

from typing import Optional, Any, List, Dict, Set, Tuple
import json


def decode_round_state(encoded_round_state: dict, my_uuid: str, my_hole_card: List[str]) -> Dict[str, Any]:
    """
    Note that in the returned state, all players have no hole cards since we have no information!!!
    """
    enc = encoded_round_state

    my_hole_card_decoded: list[Card] = [Card.from_str(c) for c in my_hole_card]
    community_card_decoded: list[Card] = [Card.from_str(c) for c in enc["community_card"]]
    known_cards: Set[int] = {c.to_id() for c in community_card_decoded} \
                                | {c.to_id() for c in my_hole_card_decoded}
    deck: Deck = Deck(deck_ids=set(range(1, 53))-known_cards)
    deck.shuffle()

    tab = Table(cheat_deck=deck)
    tab.dealer_btn = enc["dealer_btn"]
    tab.set_blind_pos(enc["small_blind_pos"], enc["big_blind_pos"])
    tab._community_card = community_card_decoded

    seats = Seats()
    seat_map = {}
    for p in enc["seats"]:
        player: Player = Player(p["uuid"], p["stack"], p["name"])

        # set player's Payinfo object (payinfo.state defaults to PAY_TILL_END)
        if p["state"] == "allin":
            player.pay_info.update_to_allin()
        elif p["state"] == "folded":
            player.pay_info.update_to_fold()
        #else:
        #    player.pay_info.status = PayInfo.PAY_TILL_END

        if p['uuid'] == my_uuid:
            player.hole_card = my_hole_card_decoded

        seats.sitdown(player)
        seat_map[player.uuid] = player

    # restore current-street chip flows & histories
    for pl in seats.players:
        # make every street slot at least an empty list
        for i in range(4):
            pl.round_action_histories[i] = []
    for street_idx, (street_name, street_hist) in enumerate(enc['action_histories'].items()):
        for player_hist in street_hist:
            if player_hist is None:
                continue
            player: Player = seat_map[player_hist["uuid"]]

            paid: int = 0
            if player_hist['action'] in (Player.ACTION_CALL_STR, Player.ACTION_RAISE_STR):
                paid = player_hist['paid']
            elif player_hist['action'] in (Player.ACTION_BIG_BLIND, Player.ACTION_SMALL_BLIND):
                paid = player_hist['amount']
            else:
                #assert player_hist['action'] == Player.ACTION_FOLD_STR
                pass
            player.pay_info.update_by_pay(paid)
            if street_name == enc["street"]:
                player.action_histories.append(player_hist)
            else:
                player.round_action_histories[street_idx].append(player_hist)

    tab.seats = seats

    return {
        "round_count": enc["round_count"],
        "small_blind_amount": enc["small_blind_amount"],
        "street": {
            "preflop":  Const.Street.PREFLOP,
            "flop":     Const.Street.FLOP,
            "turn":     Const.Street.TURN,
            "river":    Const.Street.RIVER,
            "showdown": Const.Street.SHOWDOWN,
        }[enc["street"]],
        "next_player": enc["next_player"],
        "table": tab,
    }





class UCTPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.my_index: Optional[int] = None
        self.round_start_stacks: Optional[List[int]] = None

    def declare_action(self, valid_actions, hole_card, round_state) -> Tuple[str, int]:
        # valid_actions format => [fold_action_info, call_action_info, raise_action_info]
        #print(valid_actions)
        decoded_round_state: Dict[str, Any] = decode_round_state(round_state, self.uuid, hole_card)

        if Tunables.MONTE_CARLO:
            action, amount = UCT.search(
                root_state=decoded_round_state,
                my_index=self.my_index,
                round_start_stacks=self.round_start_stacks,
                time_limit=9.5,
            )
        elif Tunables.USE_PROB_AGENT:
            action, amount = ProbAgent.act(
                sim_state=decoded_round_state,
                player_index=self.my_index,
                actions=valid_actions,
                round_start_stacks=self.round_start_stacks,
                try_all_raise_values=True
            )
        else:
            # always allin
            if valid_actions[2]['amount']['max'] != -1:
                action, amount = 'raise', valid_actions[2]['amount']['max']
            else:
                action, amount = 'call', valid_actions[1]['amount']



        #assert (
        #    (action == valid_actions[0]['action'] and amount == valid_actions[0]['amount'])
        #    or (action == valid_actions[1]['action'] and amount == valid_actions[1]['amount'])
        #    or (action == valid_actions[2]['action'] and valid_actions[2]['amount']['min'] <= amount <= valid_actions[2]['amount']['max'])
        #), f"action: {action}, amount: {amount}, valid_actions: {valid_actions}"

        return action, amount  # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.my_index = next(i for i,seat in enumerate(seats) if seat["uuid"] == self.uuid)
        self.round_start_stacks = [s["stack"] for s in seats]

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass



def setup_ai():
    return UCTPlayer()
