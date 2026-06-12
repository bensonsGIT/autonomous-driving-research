ENV_CONFIG = {
    "action": {"type": "ContinuousAction"},
    "lanes_count": 4,
    "vehicles_count": 50,
    "duration": 40,            
    "policy_frequency": 1,   
    "normalize_reward": True,
}

MAX_STEPS = ENV_CONFIG["duration"] * ENV_CONFIG["policy_frequency"]
