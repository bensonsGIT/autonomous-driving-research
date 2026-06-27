ENV_CONFIG = {
    "action": {"type": "ContinuousAction"},
    "lanes_count": 4,
    "vehicles_count": 25,
    "duration": 400,
    "policy_frequency": 10,
    "normalize_reward": True,
}

MAX_STEPS = ENV_CONFIG["duration"] * ENV_CONFIG["policy_frequency"]
