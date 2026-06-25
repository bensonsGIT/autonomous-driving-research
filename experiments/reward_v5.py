def compute_reward(obs, action, info):
    """
    Current reward function — modified each iteration by the optimization loop.
    See rewards/baseline.py for the original v0 reference (do not modify that file).
    """
    reward = 0.0
    speed = info.get("speed", 0)
    crashed = info.get("crashed", False)
    on_road = info.get("on_road", True)
    
    # Moderate crash penalty - very high penalties can hurt exploration
    # Reduced from v4's -20 which caused worse crash rate
    if crashed:
        reward -= 15.0
        return reward
    
    # Stronger off-road penalty to keep agent on track
    if not on_road:
        reward -= 4.0
        return reward
    
    # New speed reward structure with much stronger penalties for excessive speed
    # Target: 20-28 m/s (72-100 km/h) for safe highway driving
    optimal_speed_low = 20.0
    optimal_speed_high = 28.0
    danger_speed = 32.0
    
    if speed < optimal_speed_low:
        # Linear reward up to optimal range
        reward += (speed / optimal_speed_low) * 0.8
    elif speed <= optimal_speed_high:
        # Full reward in optimal range - sweet spot for safe driving
        reward += 0.8
    elif speed <= danger_speed:
        # Linear penalty as speed increases beyond optimal
        excess = speed - optimal_speed_high
        penalty_factor = excess / (danger_speed - optimal_speed_high)
        reward += 0.8 * (1.0 - penalty_factor)
    else:
        # Strong quadratic penalty for dangerous speeds
        excess = speed - danger_speed
        # At 36 m/s: penalty = -0.64, at 40 m/s: penalty = -1.28
        reward -= (excess * excess) * 0.04
    
    # Moderate survival bonus - not too high to avoid overshadowing speed control
    reward += 0.4
    
    # Small bonus for maintaining good minimum speed
    if 15.0 <= speed <= 30.0:
        reward += 0.15
    
    # Penalize steering more to encourage stability and reduce crashes
    try:
        if hasattr(action, '__len__') and len(action) > 0:
            steering_action = action[0]
            # Quadratic penalty for aggressive steering
            reward -= abs(steering_action) * abs(steering_action) * 0.08
        elif hasattr(action, 'item'):
            steering_action = action.item()
            reward -= abs(steering_action) * abs(steering_action) * 0.08
        else:
            steering_action = float(action)
            reward -= abs(steering_action) * abs(steering_action) * 0.08
    except (IndexError, TypeError, ValueError):
        pass
    
    # Enhanced proximity-based safety with earlier warnings
    try:
        # obs shape is (5, 5): [presence, x, y, vx, vy] for ego + 4 neighbors
        collision_risk = 0.0
        safe_vehicles = 0
        very_close = 0
        
        for i in range(1, min(5, len(obs))):
            if obs[i][0] > 0.5:  # Vehicle is present
                x_dist = abs(obs[i][1])
                y_dist = abs(obs[i][2])
                
                # Combined distance metric - prioritize staying away
                combined_dist = (x_dist * x_dist) + (y_dist * y_dist * 0.5)
                
                if combined_dist < 0.01:
                    # Imminent collision - very strong penalty
                    collision_risk += 0.5
                    very_close += 1
                elif combined_dist < 0.025:
                    # Critical danger zone
                    collision_risk += 0.3
                    very_close += 1
                elif combined_dist < 0.05:
                    # High risk zone
                    collision_risk += 0.15
                elif combined_dist < 0.08:
                    # Medium risk zone - early warning
                    collision_risk += 0.05
                elif combined_dist > 0.15:
                    # Safe distance maintained
                    safe_vehicles += 1
        
        # Apply collision risk penalties
        reward -= collision_risk
        
        # Reward maintaining safe distances
        reward += safe_vehicles * 0.08
        
        # Extra penalty for being surrounded
        if very_close >= 2:
            reward -= 0.3
        elif very_close >= 3:
            reward -= 0.5
        
    except (IndexError, TypeError, AttributeError):
        pass
    
    # Encourage consistent forward progress
    try:
        if len(obs) > 0 and len(obs[0]) > 3:
            vx = obs[0][3]
            # Stronger reward for good forward velocity
            if vx > 0.7:
                reward += 0.15
            elif vx > 0.5:
                reward += 0.08
    except (IndexError, TypeError, AttributeError):
        pass
    
    # Bonus for smooth acceleration control
    try:
        if hasattr(action, '__len__') and len(action) > 1:
            accel_action = action[1]
            # Small penalty for extreme acceleration/braking
            if abs(accel_action) > 0.7:
                reward -= 0.05
    except (IndexError, TypeError, ValueError):
        pass
    
    return reward
