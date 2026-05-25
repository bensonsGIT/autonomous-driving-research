def compute_reward(obs, action, info):
    """
    Inputs:
      obs    — numpy array of shape (vehicles, features)
               features: [presence, x, y, vx, vy, cos_h, sin_h]
      action — [steering, acceleration] for continuous action space
      info   — dict with keys: speed, crashed, rewards, cost
    Returns:
      float reward
    """
    reward = 0.0

    # 1. Reward forward speed (normalized 0-1 by max speed)
    speed = info.get("speed", 0)
    reward += speed / 30.0 * 1.0

    # 2. Hard penalty for crashing
    if info.get("crashed", False):
        reward -= 5.0

    # 3. Small penalty for harsh steering
    reward -= abs(action[0]) * 0.1

    return reward