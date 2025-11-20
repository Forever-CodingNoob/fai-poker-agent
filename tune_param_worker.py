from game.game import setup_config, start_poker
from agents.config import Tunables
import os, json, random
import optuna
from statistics import geometric_mean

from baseline1 import setup_ai as baseline1_ai
from baseline2 import setup_ai as baseline2_ai
from baseline3 import setup_ai as baseline3_ai
from baseline4 import setup_ai as baseline4_ai
from baseline5 import setup_ai as baseline5_ai
from baseline6 import setup_ai as baseline6_ai
from baseline7 import setup_ai as baseline7_ai
from agents.my_player import setup_ai as player_ai

def play_match(p1_algo, p2_algo, rounds=20):
    config = setup_config(
        max_round=rounds,
        initial_stack=1000,
        small_blind_amount=5
    )
    config.register_player(name="p1", algorithm=p1_algo)
    config.register_player(name="p2", algorithm=p2_algo)
    result = start_poker(config, verbose=0)
    seats = {p["name"]: p["stack"] for p in result["players"]}
    return seats["p1"], seats["p2"]



def score_and_detail(p1_factory, p2_factory):
    matches = []
    p2_win_count = 0

    score = []
    for i in range(5):
        if random.getrandbits(1):
            p1_s, p2_s = play_match(p1_factory(), p2_factory())
        else:
            p2_s, p1_s = play_match(p2_factory(), p1_factory())
        did_win = (p2_s > p1_s)
        matches.append({
            "p1_stack": p1_s,
            "p2_stack": p2_s,
            "p2_win": did_win
        })
        score.append(p2_s / 1000.0 if p2_s>0 else 1/1000)

    mean = geometric_mean(score)
    return matches, mean

# ===================================

baselines = [
    ("baseline4", baseline4_ai),
    ("baseline5", baseline5_ai),
    ("baseline6", baseline6_ai),
    ("baseline7", baseline7_ai),
]

def objective(trial):
    Tunables.MU_ROUND           = trial.suggest_float("mu_round", 1.0, 10.0, step=0.1)
    Tunables.SIGMA_ROUND        = trial.suggest_float("sigma_round", 5.0, 30.0, step=0.1)
    Tunables.UCB_C  = trial.suggest_float("ucb_c", 0.5, 4, step=0.1)
    #Tunables.UCB_DYN_C          = trial.suggest_categorical("ucb_dyn_c", [True, False])
    #Tunables.UCB_DYN_C_K        = trial.suggest_float("ucb_dyn_c_k", 0.1, 0.8, step=0.01)
    #Tunables.UCB_DYN_C_CAP      = trial.suggest_int("ucb_dyn_c_cap", 2000, 10000)
    Tunables.SMOOTH_GAMMA       = trial.suggest_float("smooth_gamma", 0.02, 1, step=0.02)
    Tunables.SMOOTH_ETA         = trial.suggest_float("smooth_eta", 0.1, 2.0, step=0.1)
    #Tunables.SMOOTH_D           = trial.suggest_float("smooth_d", 0.00003, 0.002+0.00001, step=0.00002)
    #Tunables.MAX_RAISES_PER_NODE= trial.suggest_int("max_raise", 1, 20)
    #Tunables.LOSE_PENALTY       = trial.suggest_float("lose_penalty", 0, 10.0, step=0.1)
    #Tunables.ALLOW_RAISE_MAX    = trial.suggest_categorical("allow_raise_max", [True, False])
    #Tunables.RAISE_GAP          = trial.suggest_categorical("raise_gap", ['linear', 'log'])
    #Tunables.UCT_SMOOTH         = trial.suggest_categorical("uct_smooth", [True, False])
    #Tunables.RAISE_LIMITER = trial.suggest_float("raise_limiter", 0.2, 1.0, step=0.01)

    total_score = 0
    for name, baseline_ai in baselines:
        _, score = score_and_detail(baseline_ai, player_ai)
        total_score += score
    return total_score

if __name__ == '__main__':
    study = optuna.load_study(
        storage="sqlite:///holdem.db",
        study_name="holdem",
    )
    study.optimize(objective, n_trials=50, n_jobs=1)
