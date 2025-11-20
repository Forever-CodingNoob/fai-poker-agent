import json, sys, random
from game.game import setup_config, start_poker
from matplotlib import pyplot as plt
import numpy as np
from baseline1 import setup_ai as baseline1_ai
from baseline2 import setup_ai as baseline2_ai
from baseline3 import setup_ai as baseline3_ai
from baseline4 import setup_ai as baseline4_ai
from baseline5 import setup_ai as baseline5_ai
from baseline6 import setup_ai as baseline6_ai
from baseline7 import setup_ai as baseline7_ai
from agents.my_player import setup_ai as player_ai

from multiprocessing import Pool, cpu_count

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
        if did_win:
            p2_win_count += 1

    if p2_win_count >= 3:
        score = 5.0
    else:
        sorted_matches = sorted(matches, key=lambda m: m["p2_stack"], reverse=True)
        score = 0.0
        for m in sorted_matches[:2]:
            if m["p2_win"]:
                score += 1.5
            else:
                if m["p2_stack"] >= 500:
                    score += round((m["p2_stack"] / 1000.0), 1)
    return matches, score

def worker_baseline(args):
    name, p1_factory = args
    matches, score = score_and_detail(p1_factory, player_ai)
    return name, matches, score

def main() -> dict:
    baselines = [
        ("baseline1", baseline1_ai),
        ("baseline2", baseline2_ai),
        ("baseline3", baseline3_ai),
        ("baseline4", baseline4_ai),
        ("baseline5", baseline5_ai),
        ("baseline6", baseline6_ai),
        ("baseline7", baseline7_ai),
    ]

    results = {}
    total_score = 0.0

    for name, ai in baselines:
        name, _, score = worker_baseline((name, ai))
        total_score += score
        results[name] = score

    results["total_score"] = total_score
    return results

def run_main_wrapper(_):
    return main()

def run_multiple_mains(n_runs=10):
    print(f"Running {n_runs} instances of the main function in parallel...")
    with Pool() as pool:
        all_results = pool.map(run_main_wrapper, range(n_runs))


    # Aggregate scores per baseline
    baseline_scores = dict()
    total_scores = []
    for result in all_results:
        for name, score in result.items():
            if name != "total_score":
                if name not in baseline_scores:
                    baseline_scores[name] = []
                baseline_scores[name].append(score)
            else:
                total_scores.append(score)

    print("Aggregated Results:", baseline_scores)
    # Plotting each baseline's score distribution
    num_baselines = len(baseline_scores)
    fig, axes = plt.subplots(nrows=num_baselines, ncols=1, figsize=(8, 2 * num_baselines), sharex=True)

    for ax, (name, scores) in zip(axes, baseline_scores.items()):
        scores = np.array(scores)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        ax.hist(scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(mean_score, color='red', linestyle='--', label=f"Mean: {mean_score:.2f}")
        ax.axvline(mean_score + std_score, color='green', linestyle=':', label=f"Std: ±{std_score:.2f}")
        ax.axvline(mean_score - std_score, color='green', linestyle=':')
        ax.set_title(f"{name} Score Distribution")
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        ax.legend()

    plt.tight_layout()
    plt.suptitle("Baseline AI Score Histograms", y=1.02, fontsize=14)
    plt.subplots_adjust(top=0.95)
    plt.savefig("score/baseline_scores_distribution.png")

    # Plotting total scores
    plt.figure(figsize=(8, 4))
    mean_total = np.mean(total_scores)
    std_total = np.std(total_scores)
    plt.hist(total_scores, bins=10, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(mean_total, color='red', linestyle='--', label=f"Mean: {mean_total:.2f}")
    plt.axvline(mean_total + std_total, color='green', linestyle=':', label=f"Std: ±{std_total:.2f}")
    plt.axvline(mean_total - std_total, color='green', linestyle=':')
    plt.title("Total Score Distribution")
    plt.xlabel("Total Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("score/total_scores_distribution.png")




if __name__ == "__main__":
    if len(sys.argv) == 2:
        result = main()
        print(json.dumps(result, indent=4))
        exit(0)
    run_multiple_mains(n_runs=20)
