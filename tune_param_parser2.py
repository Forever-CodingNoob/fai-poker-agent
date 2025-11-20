import optuna

def current_best(study):
    """
    Return (trial, values) for the best COMPLETE trial so far.
    If none exist yet, return None.
    Works for single- or multi-objective.
    """
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        return None          # nothing finished yet
    # for single objective      best = min() or max() depending on study.direction
    if len(study.directions) == 1:
        best = min(complete, key=lambda t: t.value) if study.directions[0].name == "MINIMIZE" \
               else max(complete, key=lambda t: t.value)
    else:  # multi-objective → pick by first objective (or implement Pareto pick)
        idx = 0
        best = min(complete, key=lambda t: t.values[idx]) if study.directions[idx].name == "MINIMIZE" \
               else max(complete, key=lambda t: t.values[idx])
    return best, (best.value if len(study.directions)==1 else best.values)


STUDY_NAME = "holdem"
STORAGE    = "sqlite:///holdem.db"   # path you used for workers

study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)

result = current_best(study)
if result is None:
    print("No completed trials yet – workers still running.")
else:
    best_trial, best_val = result
    print("Best value so far:", best_val)
    print("Best params so far:", best_trial.params)
