"""
Retrains the baseline reward function under the fixed env settings (policy_frequency=10,
duration=400, ent_coef=0.01, 500k steps). Saves results to results/baseline_v2/.

Uses the reward from experiments/reward_v1.py (labeled "v0 baseline, do not modify").
Run: python train_baseline_v2.py
"""
import importlib.util
import json
import os
import sys

import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim.env_config import ENV_CONFIG
from agents.ppo_agent import PPO_HYPERPARAMS

TRAIN_STEPS = 500_000
EVAL_EPISODES = 100
SEEDS = [42, 7, 0]
OUT_DIR = "results/baseline_v2"
MODEL_PREFIX = "agents/ppo_highway_baseline_v2"

# Load baseline reward from experiments/reward_v1.py
_spec = importlib.util.spec_from_file_location("baseline_reward", "experiments/reward_v1.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
baseline_compute_reward = _mod.compute_reward


class BaselineRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, _, done, truncated, info = self.env.step(action)
        return obs, baseline_compute_reward(obs, action, info), done, truncated, info


def make_env(render_mode=None):
    env = gym.make("highway-v0", render_mode=render_mode, config=ENV_CONFIG)
    return BaselineRewardWrapper(env)


def make_vec(n_envs=6):
    return make_vec_env(
        lambda: BaselineRewardWrapper(gym.make("highway-v0", config=ENV_CONFIG)),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
    )


def train_and_eval(seed):
    model_path = f"{MODEL_PREFIX}_{seed}"
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_env = make_vec(n_envs=6)
    model = PPO("MlpPolicy", train_env, verbose=1, seed=seed, **PPO_HYPERPARAMS)

    print(f"\n[seed {seed}] Training {TRAIN_STEPS:,} steps...")
    model.learn(total_timesteps=TRAIN_STEPS)
    model.save(model_path)
    train_env.close()
    print(f"[seed {seed}] Model saved to {model_path}.zip")

    print(f"[seed {seed}] Evaluating over {EVAL_EPISODES} episodes...")
    eval_env = make_env()
    obs, _ = eval_env.reset(seed=seed)

    ep_rewards, ep_lengths, ep_crashes, all_speeds = [], [], [], []
    cur_reward, cur_length, cur_crashed, done_count = 0, 0, False, 0

    while done_count < EVAL_EPISODES:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)
        cur_reward += reward
        cur_length += 1
        all_speeds.append(info.get("speed", 0))
        if info.get("crashed", False):
            cur_crashed = True
        if done or truncated:
            ep_rewards.append(cur_reward)
            ep_lengths.append(cur_length)
            ep_crashes.append(int(cur_crashed))
            done_count += 1
            cur_reward, cur_length, cur_crashed = 0, 0, False
            obs, _ = eval_env.reset()

    eval_env.close()

    metrics = {
        "episodes": EVAL_EPISODES,
        "mean_reward": round(float(np.mean(ep_rewards)), 2),
        "mean_length": round(float(np.mean(ep_lengths)), 2),
        "total_crashes": sum(ep_crashes),
        "crash_rate": round(sum(ep_crashes) / EVAL_EPISODES, 4),
        "min_reward": round(float(np.min(ep_rewards)), 2),
        "max_reward": round(float(np.max(ep_rewards)), 2),
        "avg_speed_ms": round(float(np.mean(all_speeds)), 2),
        "avg_speed_kmh": round(float(np.mean(all_speeds)) * 3.6, 2),
        "train_steps": TRAIN_STEPS,
        "seed": seed,
        "reward_version": "v0_baseline",
        "env_policy_frequency": ENV_CONFIG["policy_frequency"],
        "env_duration": ENV_CONFIG["duration"],
        "ppo_hyperparams": PPO_HYPERPARAMS,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/results_s{seed}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[seed {seed}] Metrics saved to {out_path}")
    print(json.dumps(metrics, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(ep_rewards, color="steelblue", alpha=0.7)
    axes[0].axhline(float(np.mean(ep_rewards)), color="red", linestyle="--",
                    label=f"Mean: {metrics['mean_reward']}")
    axes[0].set_title("Reward per Episode")
    axes[0].legend()
    axes[1].plot(ep_crashes, color="tomato", alpha=0.7)
    axes[1].set_title("Crash per Episode")
    axes[2].plot(ep_lengths, color="seagreen", alpha=0.7)
    axes[2].axhline(ENV_CONFIG["duration"], color="gray", linestyle="--",
                    label=f"Max ({ENV_CONFIG['duration']})")
    axes[2].set_title("Episode Length")
    axes[2].legend()
    plt.suptitle(f"Baseline v2 (seed {seed})", fontweight="bold")
    plt.tight_layout()
    plot_path = f"{OUT_DIR}/training_plot_s{seed}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"[seed {seed}] Plot saved to {plot_path}")

    return metrics


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    all_metrics = []
    for seed in SEEDS:
        m = train_and_eval(seed)
        all_metrics.append(m)

    avg = {
        "mean_reward": round(np.mean([m["mean_reward"] for m in all_metrics]), 2),
        "mean_length": round(np.mean([m["mean_length"] for m in all_metrics]), 2),
        "crash_rate": round(np.mean([m["crash_rate"] for m in all_metrics]), 4),
        "avg_speed_kmh": round(np.mean([m["avg_speed_kmh"] for m in all_metrics]), 2),
        "seeds": SEEDS,
        "reward_version": "v0_baseline",
        "env_policy_frequency": ENV_CONFIG["policy_frequency"],
        "train_steps": TRAIN_STEPS,
    }
    avg_path = f"{OUT_DIR}/avg_results.json"
    with open(avg_path, "w") as f:
        json.dump(avg, f, indent=2)
    print(f"\nAveraged across seeds → {avg_path}")
    print(json.dumps(avg, indent=2))
