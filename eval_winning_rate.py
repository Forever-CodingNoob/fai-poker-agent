import multiprocessing as mp
import numpy as np
import random, csv, itertools
from game.game import setup_config, start_poker
from agents.config import Tunables
from agents.my_player import setup_ai as player_ai

# ------------ import AIs -------------------------------------------------
from baseline0 import setup_ai as baseline0_ai
from baseline1 import setup_ai as baseline1_ai
from baseline2 import setup_ai as baseline2_ai
from baseline3 import setup_ai as baseline3_ai
from baseline4 import setup_ai as baseline4_ai
from baseline5 import setup_ai as baseline5_ai
from baseline6 import setup_ai as baseline6_ai
from baseline7 import setup_ai as baseline7_ai

# -------------------------------------------------------------------------

baselines = [
    ("baseline0", baseline0_ai),
    ("baseline1", baseline1_ai),
    ("baseline2", baseline2_ai),
    ("baseline3", baseline3_ai),
    ("baseline4", baseline4_ai),
    ("baseline5", baseline5_ai),
    ("baseline6", baseline6_ai),
    ("baseline7", baseline7_ai),
]

# 9 tuned parameter dictionaries (one per agent)
AGENTS = [
    ('basic', dict(UCB_DYN_C=False, UCT_SMOOTH=False, USE_PROB_AGENT=False, MONTE_CARLO=True)),
    ('dynC', dict(UCB_DYN_C=True, UCT_SMOOTH=False, USE_PROB_AGENT=False, MONTE_CARLO=True)),
    ('smoothUCT', dict(UCB_DYN_C=False, UCT_SMOOTH=True, USE_PROB_AGENT=False, MONTE_CARLO=True)),
    ('dynC+smoothUCT', dict(UCB_DYN_C=True, UCT_SMOOTH=True, USE_PROB_AGENT=False, MONTE_CARLO=True)),
    ('PBagent', dict(UCB_DYN_C=False, UCT_SMOOTH=False, USE_PROB_AGENT=True, MONTE_CARLO=True)),
    ('dynC+PBagent', dict(UCB_DYN_C=True, UCT_SMOOTH=False, USE_PROB_AGENT=True, MONTE_CARLO=True)),
    ('smoothUCT+PBagent', dict(UCB_DYN_C=False, UCT_SMOOTH=True, USE_PROB_AGENT=True, MONTE_CARLO=True)),
    ('dynC+smoothUCT+PBagent', dict(UCB_DYN_C=True, UCT_SMOOTH=True, USE_PROB_AGENT=True, MONTE_CARLO=True)),
    ('PBagent w/o MCTS', dict(UCB_DYN_C=False, UCT_SMOOTH=False, USE_PROB_AGENT=True, MONTE_CARLO=False))
]


# -------------------------------------------------------------------------

def play_match(p1_algo, p2_algo, rounds=20, seed=None):
    """Return final stacks (p1, p2)."""
    if seed is not None:
        random.seed(seed)
    cfg = setup_config(max_round=rounds, initial_stack=1000, small_blind_amount=5)
    cfg.register_player(name="p1", algorithm=p1_algo)
    cfg.register_player(name="p2", algorithm=p2_algo)
    res = start_poker(cfg, verbose=0)
    seats = {p["name"]: p["stack"] for p in res["players"]}
    return seats["p1"], seats["p2"]


def one_job(job):
    params, baseline_factory, swap, seed, a_idx, b_name = job

    # --- set global Tunables ---
    for k, v in params.items():
        setattr(Tunables, k.upper(), v)      # keys are lower-case in dict

    if swap:
        p1, p2 = play_match(baseline_factory(), player_ai(), seed=seed)
        win = int(p2 > p1)
    else:
        p1, p2 = play_match(player_ai(), baseline_factory(), seed=seed)
        win = int(p1 > p2)

    return a_idx, b_name, win
# -------------------------------------------------------------------------

def build_joblist(n_matches=100):
    jobs = []
    for a_idx, (a_name, param_dict) in enumerate(AGENTS):
        for b_name, b_fac in baselines:
            for k in range(n_matches):
                swap = bool(k & 1)
                seed = (hash((a_name, b_name, k)) & 0xFFFFFFFF)
                jobs.append( (param_dict, b_fac, swap, seed, a_idx, b_name) )
    return jobs

def main(n_matches=100):
    jobs = build_joblist(n_matches)
    # run parallel
    with mp.Pool() as pool:
        results = pool.map(one_job, jobs, chunksize=32)

    win_mat = np.zeros((len(AGENTS), len(baselines)), dtype=int)
    for a_idx, b_name, win in results:
        b_idx = [idx for idx,(n,_) in enumerate(baselines) if n==b_name][0]
        win_mat[a_idx, b_idx] += win

    win_rates = win_mat / n_matches     # 9×8 float array

    # -------- pretty print -------------
    hdr = ["agent \\ baseline"] + [b[0] for b in baselines]
    widths = [max(len(h),7) for h in hdr]

    row_fmt = "  ".join(f"{{:<{w}}}" for w in widths)

    print(row_fmt.format(*hdr))
    for (a_name,_), row in zip(AGENTS, win_rates):
        cells = [f"{x:.3f}" for x in row]
        print(row_fmt.format(a_name, *cells))

    # save as CSV
    with open("score/win_rates.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(hdr)
        for (a_name,_), row in zip(AGENTS, win_rates):
            writer.writerow([a_name] + list(row))

if __name__ == "__main__":
    main(n_matches=100)

