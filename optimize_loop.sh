#!/bin/bash
# Agentic reward optimization loop.
# Each iteration: train → evaluate → commit → agent proposes next reward.

set -e

ITERATION=${1:-1}
MAX_ITER=${2:-10}

while [ "$ITERATION" -le "$MAX_ITER" ]; do
    echo "===== Iteration $ITERATION / $MAX_ITER ====="

    # Run training + evaluation
    python train.py

    # Save reward version
    cp rewards/reward_fn.py "experiments/reward_v${ITERATION}.py"

    # Copy metrics
    cp results/latest.json "experiments/metrics_v${ITERATION}.json"

    # Commit this iteration
    git add rewards/reward_fn.py experiments/ results/latest.json
    git commit -m "iter ${ITERATION}: agent reward update"
    git tag "v${ITERATION}"

    echo "Iteration $ITERATION committed and tagged."

    # Let mini-swe-agent propose next reward (it writes to rewards/reward_fn.py)
    # Uncomment when agent integration is ready:
    # python agents/run_optimizer.py --iteration "$ITERATION"

    ITERATION=$((ITERATION + 1))
done

echo "Optimization loop complete."