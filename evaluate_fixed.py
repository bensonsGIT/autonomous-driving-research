"""
Fixed evaluation harness — behavior metrics only, reward-function-agnostic.
Run after each training iteration to produce cross-iteration-comparable metrics.
"""
import argparse
import json
import os
import sys
import numpy as np
import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from sim.env_config import ENV_CONFIG

EVAL_EPISODES = 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    SEED = args.seed
    MODEL_PATH = f"agents/ppo_highway_opt_{SEED}"
    OUTPUT_PATH = f"results/fixed_metrics_seed_{SEED}.json"

    env = gym.make("highway-v0", config=ENV_CONFIG)

    if not (os.path.exists(MODEL_PATH) or os.path.exists(MODEL_PATH + ".zip")):
        print(f"[fixed_eval] No model at {MODEL_PATH}, skipping.")
        sys.exit(1)

    model = PPO.load(MODEL_PATH, env=env)
    obs, _ = env.reset(seed=SEED)


    episode_lengths = []
    episode_crashes = []
    all_speeds = []
    current_length = 0
    current_crashed = False
    episodes_done = 0

    while episodes_done < EVAL_EPISODES:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, info = env.step(action)

        current_length += 1
        all_speeds.append(info.get("speed", 0))
        if info.get("crashed", False):
            current_crashed = True

        if done or truncated:
            episode_lengths.append(current_length)
            episode_crashes.append(int(current_crashed))
            episodes_done += 1
            current_length = 0
            current_crashed = False
            obs, _ = env.reset()

    env.close()

    metrics = {
        "episodes": EVAL_EPISODES,
        "crash_rate": round(sum(episode_crashes) / EVAL_EPISODES, 4),
        "total_crashes": sum(episode_crashes),
        "mean_length": round(float(np.mean(episode_lengths)), 2),
        "avg_speed_ms": round(float(np.mean(all_speeds)), 2),
        "avg_speed_kmh": round(float(np.mean(all_speeds)) * 3.6, 2),
        "seed": SEED,
    }

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[fixed_eval] seed={SEED}  crash_rate={metrics['crash_rate']:.2%}  "
          f"avg_speed={metrics['avg_speed_kmh']} km/h  "
          f"mean_length={metrics['mean_length']}")


if __name__ == "__main__":
    main()
