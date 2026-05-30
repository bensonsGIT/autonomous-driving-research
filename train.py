import json
import os
import numpy as np
from agents.ppo_agent import make_env, load_or_create_model

TRAIN_STEPS = 5_000
EVAL_STEPS = 10_000
MODEL_PATH = "agents/ppo_highway"


def main():
    env = make_env()
    model = load_or_create_model(env)

    print(f"Training for {TRAIN_STEPS} steps...")
    model.learn(total_timesteps=TRAIN_STEPS)
    model.save(MODEL_PATH)
    print("Model saved.")

    print(f"Evaluating for {EVAL_STEPS} steps...")
    obs, _ = env.reset()
    total_reward, crashes, steps = 0, 0, 0
    speeds = []

    while steps < EVAL_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        crashes += int(info.get("crashed", False))
        speeds.append(info.get("speed", 0))
        steps += 1
        if done or truncated:
            obs, _ = env.reset()

    env.close()

    os.makedirs("results", exist_ok=True)
    metrics = {
        "total_reward": round(total_reward, 2),
        "crashes": crashes,
        "steps": steps,
        "reward_per_step": round(total_reward / steps, 4),
        "avg_speed_ms": round(np.mean(speeds), 2),
        "avg_speed_kmh": round(np.mean(speeds) * 3.6, 2),
    }
    with open("results/latest.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()