"""
LLM-based reward function optimizer.
Called by optimize_loop.sh after each training iteration.
Reads the current metrics and reward function, then proposes an improved reward.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REWARD_PATH = Path("rewards/reward_fn.py")
METRICS_DIR = Path("experiments")


def load_context(iteration: int) -> tuple[str, dict]:
    reward_code = REWARD_PATH.read_text()

    metrics_path = METRICS_DIR / f"metrics_v{iteration}.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    return reward_code, metrics


def build_prompt(reward_code: str, metrics: dict, iteration: int) -> str:
    return f"""You are optimizing a reward function for an autonomous highway driving agent (PPO, highway-env).

## Current reward function (iteration {iteration})
```python
{reward_code}
```

## Performance metrics from this iteration
```json
{json.dumps(metrics, indent=2)}
```

## Task
Write an improved reward function to `rewards/reward_fn.py`. The function signature must remain:
    def compute_reward(obs, action, info) -> float

Guidelines:
- `obs`: (5, 5) kinematic observation matrix (ego + 4 neighbors: presence, x, y, vx, vy)
- `info` keys: "speed", "crashed", "on_road", "rewards" (dict of native sub-rewards)
- `action`: continuous [steering, acceleration] in [-1, 1]
- Penalize crashes heavily, reward sustained safe high-speed driving
- Write ONLY the Python code to the file, no explanation or markdown fences
"""


def propose_reward(prompt: str) -> None:
    subprocess.run(["mini", "-t", prompt, "-y"], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()

    print(f"[optimizer] Loading context for iteration {args.iteration}...")
    reward_code, metrics = load_context(args.iteration)

    print(f"[optimizer] Crash rate: {metrics['crash_rate']:.2%}  Mean reward: {metrics['mean_reward']}")

    prompt = build_prompt(reward_code, metrics, args.iteration)
    print("[optimizer] Requesting improved reward function from agent...")
    propose_reward(prompt)

    print(f"[optimizer] reward_fn.py updated for iteration {args.iteration + 1}.")


if __name__ == "__main__":
    main()
