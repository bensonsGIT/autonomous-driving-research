import json
import os
import numpy as np
from agents.ppo_agent import make_env, load_or_create_model

TRAIN_STEPS = 1_000_000
EVAL_EPISODES = 100
MODEL_PATH = "agents/ppo_highway"

def main():
    env = make_env()
    model = load_or_create_model(env)

    # --- Training ---
    print(f"Training for {TRAIN_STEPS} steps...")
    model.learn(total_timesteps=TRAIN_STEPS)
    model.save(MODEL_PATH)
    print("Model saved.")

    # --- Episode-based evaluation ---
    print(f"Evaluating over {EVAL_EPISODES} episodes...")
    obs, _ = env.reset()

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
        obs, reward, done, truncated, info = env.step(action)

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

            # reset for next episode
            current_reward = 0
            current_length = 0
            current_crashed = False
            obs, _ = env.reset()

    env.close()
    env.close()

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
    }

    with open("results/latest.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print("Done. Run evaluate.py to watch the agent drive.")

if __name__ == "__main__":
    main()