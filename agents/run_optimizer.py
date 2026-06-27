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


def load_context(iteration: int) -> tuple[str, dict, list[dict]]:
    reward_code = REWARD_PATH.read_text()

    metrics_path = METRICS_DIR / f"fixed_metrics_v{iteration}.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    history = []
    for i in range(1, iteration):
        p = METRICS_DIR / f"fixed_metrics_v{i}.json"
        if p.exists():
            with open(p) as f:
                history.append({"iteration": i, **json.load(f)})

    return reward_code, metrics, history


def diagnose(metrics: dict, history: list[dict]) -> str:
    crash_rate = metrics["crash_rate"]
    speed_kmh = metrics["avg_speed_kmh"]
    length = metrics["mean_length"]
    max_length = 40

    lines = []

    if crash_rate == 0.0:
        lines.append("- Crash rate is 0% — safety is solved. Do NOT increase crash penalties further; the agent is already safe.")
    elif crash_rate <= 0.05:
        lines.append(f"- Crash rate is low ({crash_rate:.1%}). Modest crash penalty is sufficient; focus on speed.")
    else:
        lines.append(f"- Crash rate is high ({crash_rate:.1%}). Prioritize safety before optimizing speed.")

    if speed_kmh < 60:
        lines.append(f"- Speed is low ({speed_kmh} km/h). Reward function is too conservative — increase speed incentives.")
    elif speed_kmh < 80:
        lines.append(f"- Speed is moderate ({speed_kmh} km/h). Push the agent toward higher sustained speeds.")
    else:
        lines.append(f"- Speed is good ({speed_kmh} km/h). Fine-tune to push further without inducing crashes.")

    if length >= max_length * 0.95:
        lines.append(f"- Episode length is maxed out ({length}/{max_length}). Agent survives full episodes consistently.")
    else:
        lines.append(f"- Episode length is {length}/{max_length}. Agent is crashing or going off-road before episode ends.")

    if history:
        prev_speed = history[-1]["avg_speed_kmh"]
        prev_crashes = history[-1]["crash_rate"]
        speed_delta = round(speed_kmh - prev_speed, 2)
        trend = "improved" if speed_delta > 0 else "regressed"
        lines.append(f"- Speed {trend} by {abs(speed_delta)} km/h vs last iteration (crash rate: {prev_crashes:.1%} → {crash_rate:.1%}).")

    return "\n".join(lines)


def format_history(history: list[dict]) -> str:
    if not history:
        return "No previous iterations."
    rows = ["iter | crash_rate | avg_speed_kmh | mean_length"]
    rows.append("-----|------------|---------------|------------")
    for h in history:
        rows.append(f"  {h['iteration']:2d} | {h['crash_rate']:10.2%} | {h['avg_speed_kmh']:13.1f} | {h['mean_length']:11.1f}")
    return "\n".join(rows)


def build_prompt(reward_code: str, metrics: dict, history: list[dict], iteration: int) -> str:
    return f"""You are optimizing a reward function for an autonomous highway driving agent (PPO, highway-env).
Each iteration trains a fresh PPO agent for 100k steps and evaluates it in the raw environment.
Episode max length is 40 steps. Target: crash_rate=0%, avg_speed_kmh as high as possible, mean_length=40.

## Iteration history (fixed behavior metrics, reward-agnostic)
{format_history(history)}

## Current iteration ({iteration}) metrics
```json
{json.dumps(metrics, indent=2)}
```

## Diagnosis
{diagnose(metrics, history)}

## Current reward function (iteration {iteration})
```python
{reward_code}
```

## Task
Write an improved reward function to `rewards/reward_fn.py`. The function signature must remain:
    def compute_reward(obs, action, info) -> float

Guidelines:
- `obs`: (5, 5) kinematic observation matrix (ego + 4 neighbors: presence, x, y, vx, vy)
- `info` keys: "speed", "crashed", "on_road", "rewards" (dict of native sub-rewards)
- `action`: continuous [steering, acceleration] in [-1, 1]
- Write ONLY the Python code to the file, no explanation or markdown fences
"""


MINI_PATH = "/Users/benson/mini-swe-agent/.venv/bin/mini"


def propose_reward(prompt: str) -> None:
    import hashlib
    before = hashlib.md5(REWARD_PATH.read_bytes()).hexdigest()
    result = subprocess.run([MINI_PATH, "-t", prompt, "-y"], stdin=subprocess.DEVNULL)
    after = hashlib.md5(REWARD_PATH.read_bytes()).hexdigest()
    if before == after:
        raise RuntimeError("mini exited without modifying reward_fn.py")
    if result.returncode != 0:
        print(f"[optimizer] mini exited with code {result.returncode} (file was modified — continuing)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()

    print(f"[optimizer] Loading context for iteration {args.iteration}...")
    reward_code, metrics, history = load_context(args.iteration)

    print(f"[optimizer] crash_rate={metrics['crash_rate']:.2%}  avg_speed={metrics['avg_speed_kmh']} km/h  mean_length={metrics['mean_length']}")

    prompt = build_prompt(reward_code, metrics, history, args.iteration)
    print("[optimizer] Requesting improved reward function from agent...")
    propose_reward(prompt)

    print(f"[optimizer] reward_fn.py updated for iteration {args.iteration + 1}.")


if __name__ == "__main__":
    main()
