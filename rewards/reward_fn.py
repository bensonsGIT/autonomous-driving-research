def compute_reward(obs, action, info):

    reward = 0.0

    speed = info.get("speed", 0)
    reward += speed / 30.0 * 1.0

    if info.get("crashed", False):
        reward -= 5.0

    reward -= abs(action[0]) * 0.1

    return reward