import argparse
import json
import os
import numpy as np
import gymnasium as gym
import highway_env
import pygame
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from agents.ppo_agent import make_env

EVAL_EPISODES = 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="agents/ppo_highway", help="Path to model file (without .zip)")
    parser.add_argument("--out", default="results/evaluate.json", help="Path to save eval metrics JSON")
    args = parser.parse_args()

    model = PPO.load(args.model)
    print(f"Model loaded from {args.model}.")

    train_metrics = None
    train_json = args.out.replace("evaluate", "latest") if "evaluate" in args.out else None
    for path in [train_json, "results/latest.json"]:
        if path and os.path.exists(path):
            with open(path) as f:
                train_metrics = json.load(f)
            print(f"Train metrics loaded from {path}.")
            break

    pygame.init()
    env = make_env(render_mode="human")
    obs, _ = env.reset()
    clock = pygame.time.Clock()

    episode_rewards = []
    episode_lengths = []
    episode_crashes = []

    current_reward = 0
    current_length = 0
    current_crashed = False
    episodes_done = 0
    crash_flash = 0
    running = True

    while running and episodes_done < EVAL_EPISODES:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        current_reward += reward
        current_length += 1

        if info.get("crashed", False):
            current_crashed = True
            crash_flash = 10

        env.render()
        pygame.event.pump()

        surface = pygame.display.get_surface()

        if crash_flash > 0:
            flash = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            alpha = int((crash_flash / 10) * 120)
            flash.fill((255, 0, 0, alpha))
            surface.blit(flash, (0, 0))
            crash_flash -= 1

        font = pygame.font.SysFont("monospace", 20, bold=True)
        surface.blit(font.render(
            f"Episode: {episodes_done + 1}/{EVAL_EPISODES}  "
            f"Crashes: {sum(episode_crashes)}  "
            f"Reward: {current_reward:.1f}",
            True, (255, 255, 255)), (10, 10))

        pygame.display.flip()
        clock.tick(30)

        if done or truncated:
            episode_rewards.append(current_reward)
            episode_lengths.append(current_length)
            episode_crashes.append(int(current_crashed))
            episodes_done += 1

            current_reward = 0
            current_length = 0
            current_crashed = False
            obs, _ = env.reset()

    env.close()
    pygame.quit()
    print(f"Finished {episodes_done} episodes.")

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else "results", exist_ok=True)
    eval_metrics = {
        "episodes": episodes_done,
        "mean_reward": round(float(np.mean(episode_rewards)), 2) if episode_rewards else 0,
        "mean_length": round(float(np.mean(episode_lengths)), 2) if episode_rewards else 0,
        "total_crashes": sum(episode_crashes),
        "crash_rate": round(sum(episode_crashes) / max(episodes_done, 1), 4),
        "min_reward": round(float(np.min(episode_rewards)), 2) if episode_rewards else 0,
        "max_reward": round(float(np.max(episode_rewards)), 2) if episode_rewards else 0,
    }
    with open(args.out, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"Eval metrics saved to {args.out}.")
    print(json.dumps(eval_metrics, indent=2))

    if train_metrics:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        axes[0].bar(["Train", "Evaluate"],
            [train_metrics["mean_reward"], eval_metrics["mean_reward"]],
            color=["steelblue", "seagreen"])
        axes[0].set_title("Mean Reward per Episode")
        axes[0].set_ylabel("Reward")

        axes[1].bar(["Train", "Evaluate"],
            [train_metrics["crash_rate"], eval_metrics["crash_rate"]],
            color=["tomato", "tomato"])
        axes[1].set_title("Crash Rate")
        axes[1].set_ylabel("Crashes / Episode")

        axes[2].bar(["Train", "Evaluate"],
            [train_metrics["mean_length"], eval_metrics["mean_length"]],
            color=["steelblue", "seagreen"])
        axes[2].set_title("Mean Episode Length")
        axes[2].set_ylabel("Steps")
        axes[2].axhline(y=100, color="gray", linestyle="--", label="Max (100)")
        axes[2].legend()

        plt.suptitle("Autonomous Driving Agent — Train vs Evaluate", fontweight="bold")
        plt.tight_layout()
        chart_path = args.out.replace(".json", "_chart.png")
        plt.savefig(chart_path)
        plt.show()
        print(f"Chart saved to {chart_path}.")

if __name__ == "__main__":
    main()
