import os
import sys
import gymnasium as gym
import highway_env
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rewards.reward_fn import compute_reward
from sim.env_config import ENV_CONFIG


class CustomRewardWrapper(gym.Wrapper):
    """Replaces the default highway-env reward with compute_reward."""

    def __init__(self, env):
        super().__init__(env)
        self.reward_stats = {"crashes": 0, "steps": 0}

    def step(self, action):
        obs, _, done, truncated, info = self.env.step(action)

        # compute_reward can return either a float or (float, dict)
        result = compute_reward(obs, action, info)
        if isinstance(result, tuple):
            custom_reward, breakdown = result
            info["reward_breakdown"] = breakdown  # store breakdown in info for logging
        else:
            custom_reward = result

        self.reward_stats["steps"] += 1
        if info.get("crashed", False):
            self.reward_stats["crashes"] += 1

        return obs, custom_reward, done, truncated, info


def make_env(render_mode=None):
    env = gym.make("highway-v0", render_mode=render_mode, config=ENV_CONFIG)
    return CustomRewardWrapper(env)


def load_or_create_model(env, model_path="agents/ppo_highway.zip"):
    # strip .zip if passed with extension since PPO.load adds it automatically
    clean_path = model_path.replace(".zip", "")

    if os.path.exists(model_path) or os.path.exists(clean_path + ".zip"):
        print(f"Loading existing model from {model_path}...")
        return PPO.load(clean_path, env=env, verbose=1, tensorboard_log="./logs/")

    print("No existing model found. Creating new model...")
    return PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")