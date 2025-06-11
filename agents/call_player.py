from game.players import BasePokerPlayer
import json


class CallPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [fold_action_info, call_action_info, raise_action_info]
        # TODO: implement UCT (UCB applied to Trees)
        #print(valid_actions)
        #print(hole_card)
        #print("HIHIHIIHIH")
        #print(json.dumps(round_state, indent=4))
        action = valid_actions[1]['action']
        amount = valid_actions[1]['amount']
        return action, amount  # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        #print("round statrs")
        #print(round_count)
        #print(hole_card)
        #print(seats)
        pass

    def receive_street_start_message(self, street, round_state):
        #print("street starts")
        #print(json.dumps(round_state, indent=4))
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return CallPlayer()
