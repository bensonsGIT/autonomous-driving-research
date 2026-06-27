#!/bin/bash
# Agentic reward optimization loop.
# Each iteration: train → evaluate → commit → agent proposes next reward.

set -e

ITERATION=${1:-1}
MAX_ITER=${2:-10}

while [ "$ITERATION" -le "$MAX_ITER" ]; do
    echo "===== Iteration $ITERATION / $MAX_ITER ====="

    # Train and evaluate across 3 seeds
    for SEED in 42 7 0; do
        echo "  -- Seed $SEED --"
        .venv/bin/python3 train.py --seed "$SEED"
        .venv/bin/python3 evaluate_fixed.py --seed "$SEED"
    done

    # Average fixed metrics across seeds
    .venv/bin/python3 average_metrics.py --seeds 42 7 0

    # Save reward version and archived metrics
    cp rewards/reward_fn.py "experiments/reward_v${ITERATION}.py"
    cp results/fixed_metrics.json "experiments/fixed_metrics_v${ITERATION}.json"
    for SEED in 42 7 0; do
        cp "results/fixed_metrics_seed_${SEED}.json" "experiments/fixed_metrics_v${ITERATION}_seed_${SEED}.json"
        cp "results/training_plot_seed_${SEED}.png" "experiments/plot_v${ITERATION}_seed_${SEED}.png"
    done

    # Commit this iteration
    git add rewards/reward_fn.py experiments/
    git commit -m "iter ${ITERATION}: agent reward update"
    git tag -f "v${ITERATION}"

    echo "Iteration $ITERATION committed and tagged."

    # Let mini-swe-agent propose next reward (it writes to rewards/reward_fn.py)
    .venv/bin/python3 agents/run_optimizer.py --iteration "$ITERATION"

    ITERATION=$((ITERATION + 1))
done

echo "Optimization loop complete."