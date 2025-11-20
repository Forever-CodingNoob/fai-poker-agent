#!/usr/bin/env bash

# ------------------------------------------------------------
# CONFIG — edit as you like
N_WORKERS=100
DB_URI="sqlite:///holdem.db"
STUDY_NAME="holdem"
# ------------------------------------------------------------

# Create the study if it does not yet exist (no-op if it does)
optuna create-study \
    --study-name "${STUDY_NAME}" \
    --storage "${DB_URI}" \
    --direction maximize --verbose --skip-if-exists || exit 1

# Array for child PIDs
pids=()

# Trap: on Ctrl-C or kill, forward the signal to every child
cleanup() {
    echo "Caught interrupt – stopping workers..."
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    # Wait for children to exit so that the exit status is clean
    wait
    exit
}
trap cleanup INT TERM

# Launch worker processes
for ((i=1; i<=N_WORKERS; i++)); do
    python3 tune_param_worker.py &
    pids+=("$!")          # remember PID
done

# Wait for all workers
wait

# Post-processing after all workers completed
python3 tune_param_parser.py
