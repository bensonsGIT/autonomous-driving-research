import gymnasium as gym
import highway_env
import pygame
import json
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# --- Load trained model ---
model = PPO.load("agents/ppo_highway")
print("Model loaded. Replaying 10,000 step evaluation visually...")

# --- Load real metrics from train.py ---
with open("results/latest.json") as f:
    metrics = json.load(f)
print(f"Real data: {metrics}")

# --- Setup pygame ---
pygame.init()
env = gym.make("highway-v0", render_mode="human", config={
    "simulation_frequency": 15,
    "policy_frequency": 5
})
obs, _ = env.reset()
steps = 0
running = True
clock = pygame.time.Clock()

# --- Visual replay of the same 10,000 steps ---
while running and steps < 10_000:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break

    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, truncated, _ = env.step(action)
    steps += 1

    pygame.display.flip()
    clock.tick(60)

    if done or truncated:
        obs, _ = env.reset()

# --- Cleanup ---
env.close()
pygame.quit()
print(f"Visual replay finished at step {steps}.")

# --- Plot the real data from train.py ---
labels = list(metrics.keys())
values = list(metrics.values())
plt.figure(figsize=(8, 4))
plt.bar(labels, values, color=["steelblue", "tomato", "gray", "seagreen"])
plt.title(f"Real evaluation metrics ({metrics['steps']} steps)")
plt.tight_layout()
plt.savefig("results/latest_metrics.png")
plt.show()
print("Chart saved to results/latest_metrics.png")