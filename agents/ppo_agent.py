import os
import sys
import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rewards.reward_fn import compute_reward
from sim.env_config import ENV_CONFIG

PPO_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}


class CustomRewardWrapper(gym.Wrapper):
    """Replaces the default highway-env reward with compute_reward."""

    def __init__(self, env):
        super().__init__(env)
        self.reward_stats = {"crashes": 0, "steps": 0}

    def step(self, action):
        obs, _, done, truncated, info = self.env.step(action)

        result = compute_reward(obs, action, info)
        if isinstance(result, tuple):
            custom_reward, breakdown = result
            info["reward_breakdown"] = breakdown
        else:
            custom_reward = result

        self.reward_stats["steps"] += 1
        if info.get("crashed", False):
            self.reward_stats["crashes"] += 1

        return obs, custom_reward, done, truncated, info


def make_env(render_mode=None):
    env = gym.make("highway-v0", render_mode=render_mode, config=ENV_CONFIG)
    return CustomRewardWrapper(env)


def make_vec_env_parallel(n_envs=4):
    return make_vec_env(
        lambda: CustomRewardWrapper(gym.make("highway-v0", config=ENV_CONFIG)),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv
    )


def load_or_create_model(env, model_path="agents/ppo_highway.zip", seed=42):

    clean_path = model_path.replace(".zip", "")

    if os.path.exists(model_path) or os.path.exists(clean_path + ".zip"):
        print(f"Loading existing model from {model_path}...")
        return PPO.load(clean_path, env=env, verbose=1, tensorboard_log="./logs/")

    print("No existing model found. Creating new model...")
    return PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/", seed=seed, **PPO_HYPERPARAMS)
