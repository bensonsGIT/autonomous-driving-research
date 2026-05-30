import json
import gymnasium as gym
import highway_env
import pygame
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from agents.ppo_agent import make_env

EVAL_STEPS = 5_000


def main():
    model = PPO.load("agents/ppo_highway")
    print("Model loaded.")

    with open("results/latest.json") as f:
        metrics = json.load(f)
    print(f"Last run metrics: {metrics}")

    pygame.init()
    env = make_env(render_mode="human")
    obs, _ = env.reset()
    steps = 0
    clock = pygame.time.Clock()
    running = True

    while running and steps < EVAL_STEPS:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, _ = env.step(action)
        steps += 1
        env.render()
        pygame.event.pump()
        pygame.display.flip()
        clock.tick(30)
        if done or truncated:
            obs, _ = env.reset()

    env.close()
    pygame.quit()
    print(f"Finished at step {steps}.")

    labels = list(metrics.keys())
    values = list(metrics.values())
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color=["steelblue", "tomato", "gray", "seagreen", "purple", "orange"])
    plt.title(f"Evaluation metrics ({metrics['steps']} steps)")
    plt.tight_layout()
    plt.savefig("results/latest_metrics.png")
    plt.show()


if __name__ == "__main__":
    main()