import optuna

STUDY_NAME = "holdem"
STORAGE    = "sqlite:///holdem.db"   # path you used for workers

study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)

print("Best value :", study.best_value)
print("Best params:", study.best_params)
print("Best trial :", study.best_trial.number)

for t in sorted(study.trials, key=lambda tr: tr.value)[:5]:
    print(t.number, t.value, t.params)
