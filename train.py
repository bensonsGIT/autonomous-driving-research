import gymnasium as gym
import highway_env
import json, os
from stable_baselines3 import PPO

# --- Setup ---
env = gym.make("highway-v0", render_mode=None, config={
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5, 
    }
})
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")

# --- Training ---
print("Training for 10,000 steps...")
model.learn(total_timesteps=10_000)
model.save("agents/ppo_highway")
print("Model saved.")

# --- Evaluate silently for 10,000 steps (real data) ---
print("Evaluating for 10,000 steps...")
obs, _ = env.reset()
total_reward, crashes, steps = 0, 0, 0

while steps < 10_000:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    crashes += int(info.get("crashed", False))
    steps += 1
    if done or truncated:
        obs, _ = env.reset()

env.close()

# --- Save real metrics ---
os.makedirs("results", exist_ok=True)
metrics = {
    "total_reward": round(total_reward, 2),
    "crashes": crashes,
    "steps": steps,
    "reward_per_step": round(total_reward / steps, 4)
}
with open("results/latest.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Results:", metrics)
print("Done. Run evaluate.py to watch the agent replay these exact conditions.")