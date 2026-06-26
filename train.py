import json
import os
import time
import pygame
import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from agents.ppo_agent import make_env, make_vec_env_parallel, load_or_create_model, PPO_HYPERPARAMS

TRAIN_STEPS = 100_000
EVAL_EPISODES = 100
MODEL_PATH = "agents/ppo_highway_opt_42"
RENDER_EVERY_N_EPISODES = 100
SEED = 42

# Seeds: 42, 7, 102, 0

# --- Render callback ---
class RenderCallback(BaseCallback):
    def __init__(self, render_every=RENDER_EVERY_N_EPISODES, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.render_every = render_every
        self.episode_count = 0
        self.render_env = None
        self.clock = None

    def _on_training_start(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.render_env = make_env(render_mode="human")

    def _on_step(self):
        dones = self.locals.get("dones", [])
        if any(dones):
            self.episode_count += 1
            if self.episode_count % self.render_every == 0:
                print(f"  [Preview] Episode {self.episode_count}...")
                obs, _ = self.render_env.reset()
                for _ in range(40):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, done, truncated, _ = self.render_env.step(action)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return False
                    pygame.display.flip()
                    self.clock.tick(60)
                    if done or truncated:
                        break
        return True

    def _on_training_end(self):
        if self.render_env:
            self.render_env.close()
        pygame.quit()


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_env = make_vec_env_parallel(n_envs=6)
    eval_env = make_env()
    model = load_or_create_model(train_env, seed=SEED, log_dir=f"./logs/seed_{SEED}")

    # --- Training ---
    print(f"Training for {TRAIN_STEPS} steps...")
    print(f"Preview every {RENDER_EVERY_N_EPISODES} episodes.")
    # callback = RenderCallback(render_every=RENDER_EVERY_N_EPISODES)
    model.learn(total_timesteps=TRAIN_STEPS)
    model.save(MODEL_PATH)
    print("Model saved.")

    print(f"Evaluating over {EVAL_EPISODES} episodes...")
    obs, _ = eval_env.reset(seed=SEED)

    episode_rewards = []
    episode_lengths = []
    episode_crashes = []
    all_speeds = []

    current_reward = 0
    current_length = 0
    current_crashed = False
    episodes_done = 0

    while episodes_done < EVAL_EPISODES:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)

        current_reward += reward
        current_length += 1
        all_speeds.append(info.get("speed", 0))

        if info.get("crashed", False):
            current_crashed = True

        if done or truncated:
            episode_rewards.append(current_reward)
            episode_lengths.append(current_length)
            episode_crashes.append(int(current_crashed))
            episodes_done += 1

            current_reward = 0
            current_length = 0
            current_crashed = False
            obs, _ = eval_env.reset()

    train_env.close()
    eval_env.close()

    # --- Save metrics ---
    os.makedirs("results", exist_ok=True)
    metrics = {
        "episodes": EVAL_EPISODES,
        "mean_reward": round(float(np.mean(episode_rewards)), 2),
        "mean_length": round(float(np.mean(episode_lengths)), 2),
        "total_crashes": sum(episode_crashes),
        "crash_rate": round(sum(episode_crashes) / EVAL_EPISODES, 4),
        "min_reward": round(float(np.min(episode_rewards)), 2),
        "max_reward": round(float(np.max(episode_rewards)), 2),
        "avg_speed_ms": round(float(np.mean(all_speeds)), 2),
        "avg_speed_kmh": round(float(np.mean(all_speeds)) * 3.6, 2),
        "train_steps": TRAIN_STEPS,
        "seed": SEED,
        "ppo_hyperparams": PPO_HYPERPARAMS,
    }
    with open("results/latest.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(episode_rewards, color="steelblue", alpha=0.7)
    axes[0].axhline(y=metrics["mean_reward"], color="red",
                    linestyle="--", label=f'Mean: {metrics["mean_reward"]}')
    axes[0].set_title("Reward per Episode")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].legend()

    axes[1].plot(episode_crashes, color="tomato", alpha=0.7)
    axes[1].set_title("Crash per Episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Crashed (1) / Safe (0)")

    axes[2].plot(episode_lengths, color="seagreen", alpha=0.7)
    axes[2].axhline(y=40, color="gray", linestyle="--", label="Max (40)")
    axes[2].set_title("Episode Length")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Steps")
    axes[2].legend()

    plt.suptitle("Training Evaluation Results", fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/training_plot.png")
    plt.show()
    print("Plot saved to results/training_plot.png")
    print("Done. Run evaluate.py to watch the full agent replay.")


if __name__ == "__main__":
    main()
