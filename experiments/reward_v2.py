def compute_reward(obs, action, info):
    """
    Improved reward function (v2) - Dramatically prioritize safety over speed.
    
    Key changes from v1:
    1. Exponentially stronger crash penalty
    2. More conservative target speed (25-30 m/s = 90-108 km/h)
    3. Progressive rewards for episode length (survival)
    4. Stronger proximity penalties to prevent aggressive lane changes
    5. Acceleration penalty to prevent aggressive driving
    6. More nuanced speed shaping to discourage excessive speed
    """
    reward = 0.0
    speed = info.get("speed", 0)
    crashed = info.get("crashed", False)
    on_road = info.get("on_road", True)
    
    # === ULTIMATE SAFETY: Extreme crash penalty ===
    if crashed:
        reward -= 200.0
        return reward
    
    # === OFF-ROAD PENALTY: Must stay on road ===
    if not on_road:
        reward -= 10.0
        return reward
    
    # === SURVIVAL BONUS: Strong incentive to complete episodes ===
    # This is the foundation - every step alive matters
    reward += 1.0
    
    # === SPEED REWARD: Conservative target for safer driving ===
    # Lower target speeds reduce crash risk
    # Target: 25-30 m/s (90-108 km/h) - highway speeds but safe
    target_speed_low = 24.0   # ~86 km/h
    target_speed_optimal = 28.0  # ~101 km/h
    target_speed_high = 32.0  # ~115 km/h
    
    if speed < 10.0:
        # Too slow - minimal reward
        reward += speed * 0.05
    elif speed < target_speed_low:
        # Below target - encourage acceleration but not too much
        reward += 0.5 + (speed - 10.0) / (target_speed_low - 10.0) * 0.5
    elif speed <= target_speed_optimal:
        # Optimal range - maximum reward
        reward += 1.5
    elif speed <= target_speed_high:
        # Acceptable but not optimal - slight reduction
        reward += 1.5 - (speed - target_speed_optimal) / (target_speed_high - target_speed_optimal) * 0.3
    else:
        # Too fast - significant penalty as crash risk increases
        excess = speed - target_speed_high
        reward += max(0.0, 1.2 - excess * 0.2)
    
    # === SMOOTH STEERING: Penalize aggressive maneuvers ===
    try:
        if hasattr(action, '__len__') and len(action) > 0:
            steering_action = action[0]
            accel_action = action[1] if len(action) > 1 else 0.0
        elif hasattr(action, 'item'):
            steering_action = action.item()
            accel_action = 0.0
        else:
            steering_action = float(action)
            accel_action = 0.0
        
        # Quadratic penalty for steering - aggressive turns are very dangerous
        steering_penalty = (abs(steering_action) ** 1.5) * 0.3
        reward -= steering_penalty
        
        # Penalize aggressive acceleration/braking
        accel_penalty = (abs(accel_action) ** 1.2) * 0.1
        reward -= accel_penalty
        
    except (IndexError, TypeError, ValueError):
        pass
    
    # === COLLISION AVOIDANCE: Strong proximity-based penalties ===
    try:
        min_distance = float('inf')
        dangerous_proximity_penalty = 0.0
        
        # Check all nearby vehicles
        for i in range(1, min(5, obs.shape[0])):
            if obs[i, 0] > 0.5:  # Vehicle is present
                x_rel = obs[i, 1]  # Relative x position (longitudinal)
                y_rel = obs[i, 2]  # Relative y position (lateral)
                vx_rel = obs[i, 3]  # Relative x velocity
                
                # Calculate Euclidean distance
                distance = (x_rel ** 2 + y_rel ** 2) ** 0.5
                min_distance = min(min_distance, distance)
                
                # Check if vehicle is ahead and closing in
                is_ahead = x_rel > 0
                is_closing = vx_rel < 0  # Relative velocity negative means closing
                
                # Extremely strong penalties for dangerous proximity
                if distance < 0.05:  # Imminent collision
                    dangerous_proximity_penalty += 2.0
                elif distance < 0.1:  # Very dangerous
                    dangerous_proximity_penalty += 1.0
                elif distance < 0.15:  # Dangerous
                    dangerous_proximity_penalty += 0.5
                elif distance < 0.25:  # Risky
                    dangerous_proximity_penalty += 0.2
                
                # Extra penalty if closing in on vehicle ahead
                if is_ahead and is_closing and distance < 0.3:
                    dangerous_proximity_penalty += 0.3
                
                # Lateral proximity penalty (discourage risky lane changes)
                if abs(y_rel) < 0.15 and abs(x_rel) < 0.3:
                    dangerous_proximity_penalty += 0.4
        
        reward -= dangerous_proximity_penalty
        
        # Bonus for maintaining safe following distance
        if 0.3 <= min_distance <= 0.6:
            reward += 0.1
        
    except (IndexError, AttributeError, TypeError):
        pass
    
    return reward
