def compute_reward(obs, action, info):
    """
    Current reward function — modified each iteration by the optimization loop.
    See rewards/baseline.py for the original v0 reference (do not modify that file).
    """
    reward = 0.0
    speed = info.get("speed", 0)
    crashed = info.get("crashed", False)
    on_road = info.get("on_road", True)
    
    # Heavily penalize crashes to reduce crash rate
    if crashed:
        reward -= 50.0
        return reward
    
    # Reward staying on road
    if not on_road:
        reward -= 5.0
        return reward
    
    # Reward high-speed driving with a more appropriate optimal speed
    # Highway driving should target ~30 m/s (108 km/h)
    optimal_speed = 30.0
    max_reward_speed = 35.0
    
    if speed < optimal_speed:
        # Linear reward up to optimal speed
        reward += (speed / optimal_speed) * 1.2
    elif speed <= max_reward_speed:
        # Full reward between optimal and max
        reward += 1.2
    else:
        # Slight penalty for excessive speed (safety concern)
        reward += 1.2 - 0.1 * (speed - max_reward_speed)
    
    # Reward survival (being alive and on road each step)
    reward += 0.15
    
    # Additional reward for maintaining good speed
    if speed > 15.0:
        reward += 0.1
    
    # Penalize excessive steering but less than before to allow lane changes
    try:
        if hasattr(action, '__len__') and len(action) > 0:
            steering_action = action[0]
            # Reduce steering penalty to allow necessary maneuvers
            reward -= abs(steering_action) * 0.015
        elif hasattr(action, 'item'):
            steering_action = action.item()
            reward -= abs(steering_action) * 0.015
        else:
            steering_action = float(action)
            reward -= abs(steering_action) * 0.015
    except (IndexError, TypeError, ValueError):
        pass
    
    # Safety bonus: reward maintaining safe distances from other vehicles
    try:
        # obs shape is (5, 5): [presence, x, y, vx, vy] for ego + 4 neighbors
        # Check neighboring vehicles (rows 1-4)
        safe_distance_bonus = 0.0
        for i in range(1, min(5, len(obs))):
            if obs[i][0] > 0.5:  # Vehicle is present
                # x is longitudinal, y is lateral distance
                x_dist = abs(obs[i][1])
                y_dist = abs(obs[i][2])
                
                # Reward safe longitudinal spacing
                if x_dist > 0.15:  # Safe distance
                    safe_distance_bonus += 0.02
                elif x_dist < 0.05:  # Too close - dangerous
                    safe_distance_bonus -= 0.05
        
        reward += safe_distance_bonus
    except (IndexError, TypeError, AttributeError):
        pass
    
    return reward
