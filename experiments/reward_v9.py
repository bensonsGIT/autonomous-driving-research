def compute_reward(obs, action, info):
    """
    V9 reward function - Breaking out of conservative local optimum.
    V8: 0% crashes, but only 17.54 m/s (too conservative, worse than V7!)
    V9: Target 21-24 m/s with <5% crash rate by making sub-optimal speeds less rewarding
    
    Key changes from V8:
    - Much steeper penalty for mediocre speeds (15-20 m/s range)
    - Stronger rewards in optimal range (20-26 m/s)
    - Better forward velocity bonuses to encourage acceleration
    - Reduced base survival bonus to make speed matter more
    - Keep strong crash penalty (V8's 0% crash rate proves it works)
    """
    reward = 0.0
    speed = info.get("speed", 0)
    crashed = info.get("crashed", False)
    on_road = info.get("on_road", True)
    
    # Strong crash penalty - V8's -20 worked well (0% crashes)
    if crashed:
        reward -= 20.0
        return reward
    
    # Strong off-road penalty
    if not on_road:
        reward -= 4.0
        return reward
    
    # MAJOR CHANGE: Much steeper speed reward curve to escape local optimum
    # Target: 20-26 m/s optimal range with strong incentive to reach it
    optimal_speed_low = 20.0
    optimal_speed_high = 26.0
    danger_speed = 32.0
    
    if speed < optimal_speed_low:
        # Steeper exponential curve with penalty for mediocre speeds
        # This breaks the local optimum at 17-18 m/s
        normalized_speed = speed / optimal_speed_low
        
        # Base exponential reward (steeper than V8)
        base_reward = (normalized_speed ** 2.0)  # More exponential
        
        # Add strong penalty for mediocre speeds (15-19 m/s)
        # Agent was getting stuck here - make it less attractive
        if speed < 19.0:
            # Penalty increases as we get further from optimal
            deficit = optimal_speed_low - speed
            mediocrity_penalty = (deficit / 10.0) * 0.4  # Up to -0.4 at 10 m/s
            base_reward -= mediocrity_penalty
        
        reward += base_reward
    elif speed <= optimal_speed_high:
        # HIGH reward in optimal range - make this very attractive
        reward += 1.5  # Increased from V8's 1.0
        # Extra bonus in sweet spot (22-24 m/s)
        if 21.5 <= speed <= 24.5:
            reward += 0.3  # Doubled from V8's 0.15
    elif speed <= danger_speed:
        # Gradual decay beyond optimal
        excess = speed - optimal_speed_high
        penalty_factor = excess / (danger_speed - optimal_speed_high)
        reward += 1.5 * (1.0 - penalty_factor * 0.7)
    else:
        # Quadratic penalty for dangerous speeds
        excess = speed - danger_speed
        reward -= (excess ** 2) * 0.05
    
    # Reduced base survival bonus - make speed matter more
    reward += 0.2  # Down from V8's 0.35
    
    # STRONGER penalties for very slow speeds - push agent to go faster
    if speed < 16.0:
        reward -= 0.45  # Increased from V8's 0.25
    elif speed < 18.0:
        reward -= 0.30  # Increased from V8's 0.12
    elif speed < 19.5:
        # NEW: penalty for being stuck in the local optimum zone
        reward -= 0.15
    
    # Steering penalty - encourage smooth driving
    try:
        if hasattr(action, '__len__') and len(action) > 0:
            steering_action = action[0]
            reward -= (steering_action ** 2) * 0.08
        elif hasattr(action, 'item'):
            steering_action = action.item()
            reward -= (steering_action ** 2) * 0.08
        else:
            steering_action = float(action)
            reward -= (steering_action ** 2) * 0.08
    except (IndexError, TypeError, ValueError):
        pass
    
    # Proximity-based safety with relative velocity awareness (keep from V8)
    try:
        collision_risk = 0.0
        safe_vehicles = 0
        very_close = 0
        ego_vx = obs[0][3] if len(obs) > 0 and len(obs[0]) > 3 else 0
        ego_vy = obs[0][4] if len(obs) > 0 and len(obs[0]) > 4 else 0
        
        for i in range(1, min(5, len(obs))):
            if obs[i][0] > 0.5:  # Vehicle is present
                x_dist = abs(obs[i][1])
                y_dist = abs(obs[i][2])
                
                # Get relative velocity
                other_vx = obs[i][3] if len(obs[i]) > 3 else 0
                other_vy = obs[i][4] if len(obs[i]) > 4 else 0
                rel_vx = abs(ego_vx - other_vx)
                rel_vy = abs(ego_vy - other_vy)
                rel_v_total = (rel_vx ** 2 + rel_vy ** 2) ** 0.5
                
                # Combined spatial distance metric
                combined_dist = (x_dist ** 2) + (y_dist ** 2 * 0.5)
                
                # Velocity-adjusted risk
                velocity_factor = min(1.0, 0.3 + rel_v_total * 2.0)
                
                if combined_dist < 0.008:
                    collision_risk += 0.6 * velocity_factor
                    very_close += 1
                elif combined_dist < 0.02:
                    collision_risk += 0.35 * velocity_factor
                    very_close += 1
                elif combined_dist < 0.045:
                    collision_risk += 0.18 * velocity_factor
                elif combined_dist < 0.08:
                    collision_risk += 0.08 * velocity_factor
                elif combined_dist > 0.15:
                    safe_vehicles += 1
        
        reward -= collision_risk
        
        # Reward maintaining safe distances
        reward += safe_vehicles * 0.12  # Slightly increased
        
        # Strong penalty for being surrounded
        if very_close >= 3:
            reward -= 0.6
        elif very_close >= 2:
            reward -= 0.35
        
    except (IndexError, TypeError, AttributeError):
        pass
    
    # ENHANCED forward velocity bonus - encourage acceleration
    try:
        if len(obs) > 0 and len(obs[0]) > 3:
            vx = obs[0][3]
            # Stronger bonuses to encourage higher speeds
            if vx > 0.75:
                reward += 0.30  # Increased from V8's 0.20
            elif vx > 0.65:
                reward += 0.22  # Increased from V8's 0.15
            elif vx > 0.55:
                reward += 0.14  # Increased from V8's 0.10
            elif vx < 0.45:
                # NEW: penalize low forward velocity
                reward -= 0.10
    except (IndexError, TypeError, AttributeError):
        pass
    
    # Acceleration control - stronger encouragement for forward acceleration
    try:
        if hasattr(action, '__len__') and len(action) > 1:
            accel_action = action[1]
            # Penalize extreme actions
            if abs(accel_action) > 0.9:
                reward -= 0.08
            # STRONGER bonus for forward acceleration
            elif accel_action > 0.4:
                reward += 0.10  # Doubled from V8's 0.05
            elif accel_action > 0.2:
                reward += 0.05
    except (IndexError, TypeError, ValueError):
        pass
    
    # ENHANCED compound bonus for optimal high-speed safe driving
    if 20.5 <= speed <= 27.0 and on_road and not crashed:
        bonus = 0.25  # Increased from V8's 0.15
        # Extra bonus if maintaining safe distances
        if safe_vehicles >= 3:
            bonus += 0.20  # Increased
        elif safe_vehicles >= 2:
            bonus += 0.12  # Increased
        reward += bonus
    
    # NEW: Progressive speed milestone bonuses
    if speed >= 22.0:
        reward += 0.15
    if speed >= 24.0:
        reward += 0.10
    
    return reward
