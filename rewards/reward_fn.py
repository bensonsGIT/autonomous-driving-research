def compute_reward(obs, action, info):
    """
    Current reward function — modified each iteration by the optimization loop.
    See rewards/baseline.py for the original v0 reference (do not modify that file).
    """
    reward = 0.0
    speed = info.get("speed", 0)
    crashed = info.get("crashed", False)

    optimal_speed = 20.0
    if speed <= optimal_speed:
        reward += (speed / optimal_speed) * 0.6
    else:
        reward += 0.6

    if crashed:
        reward -= 20.0

    if info.get("on_road", True):
        reward += 0.08

    if speed > 5.0 and not crashed:
        reward += 0.05

    try:
        if hasattr(action, '__len__') and len(action) > 0:
            steering_action = action[0]
        elif hasattr(action, 'item'):
            steering_action = action.item()
        else:
            steering_action = float(action)
        reward -= abs(steering_action) * 0.03
    except (IndexError, TypeError, ValueError):
        pass

    return reward