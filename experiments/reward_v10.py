def compute_reward(obs, action, info):
    """
    V10 reward function - Balanced approach for safe speed improvement.
    V8: 0% crashes, 17.54 m/s (too conservative)
    V9: 39% crashes, 25.93 m/s (too aggressive - major safety failure!)
    V10: Target 21-23 m/s with <5% crash rate by carefully tuning V8's conservative base
    
    Key strategy:
    - Keep V8's strong safety foundation (it achieved 0% crashes)
    - Make MODEST speed incentives (V9 went too far)
    - Increase crash penalty even more (V9's 39% crash rate shows we need stronger deterrent)
    - Maintain strict collision avoidance from V8
    - Add gradual speed incentives without removing safety rewards
    """
    reward = 0.0
    speed = info.get("speed", 0)
    crashed = info.get("crashed", False)
    on_road = info.get("on_road", True)
    
    # STRONGER crash penalty - V9's 39% crash rate shows -20 wasn't enough
    # Need to make crashes even more expensive
    if crashed:
        reward -= 25.0
        return reward
    
    # Strong off-road penalty (keep from V8)
    if not on_road:
        reward -= 4.0
        return reward
    
    # More gradual speed reward curve - V9's aggressive changes caused crashes
    # Target: 20-25 m/s optimal range (narrower than V9's 20-26)
    optimal_speed_low = 20.0
    optimal_speed_high = 25.0
    danger_speed = 30.0  # Lower threshold than V9's 32
    
    if speed < optimal_speed_low:
        # Gentler exponential curve than V9 (between V8 and V9)
        normalized_speed = speed / optimal_speed_low
        # V8 used 1.3, V9 used 2.0 - use 1.6 as middle ground
        base_reward = (normalized_speed ** 1.6)
        
        # MODEST penalty for mediocre speeds (much gentler than V9)
        if speed < 18.0:
            deficit = optimal_speed_low - speed
            mediocrity_penalty = (deficit / 10.0) * 0.15  # V9 used 0.4, too harsh
            base_reward -= mediocrity_penalty
        
        reward += base_reward
    elif speed <= optimal_speed_high:
        # Good reward in optimal range (between V8's 1.0 and V9's 1.5)
        reward += 1.2
        # Modest bonus in sweet spot
        if 21.0 <= speed <= 23.5:
            reward += 0.15  # V8 used 0.15, V9 used 0.3
    elif speed <= danger_speed:
        # Gradual decay beyond optimal
        excess = speed - optimal_speed_high
        penalty_factor = excess / (danger_speed - optimal_speed_high)
        reward += 1.2 * (1.0 - penalty_factor * 0.8)  # Steeper decay than V8
    else:
        # Strong quadratic penalty for dangerous speeds
        excess = speed - danger_speed
        reward -= (excess ** 2) * 0.08  # Stronger than V9's 0.05
    
    # Balanced survival bonus (between V8 and V9)
    reward += 0.3  # V8: 0.35, V9: 0.2
    
    # Modest penalties for slow speeds (gentler than V9)
    if speed < 16.0:
        reward -= 0.30  # V9: 0.45
    elif speed < 18.0:
        reward -= 0.15  # V9: 0.30
    elif speed < 19.0:
        reward -= 0.08  # V9: 0.15
    
    # Steering penalty - keep V8's approach (V9 didn't change this)
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
    
    # ENHANCED proximity safety - V9's 39% crashes mean we need stricter collision avoidance
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
                
                # Velocity-adjusted risk (STRONGER than V8/V9)
                velocity_factor = min(1.2, 0.4 + rel_v_total * 2.5)  # Increased sensitivity
                
                # STRICTER distance thresholds and HIGHER penalties
                if combined_dist < 0.008:
                    collision_risk += 0.8 * velocity_factor  # V9: 0.6
                    very_close += 1
                elif combined_dist < 0.02:
                    collision_risk += 0.5 * velocity_factor  # V9: 0.35
                    very_close += 1
                elif combined_dist < 0.045:
                    collision_risk += 0.25 * velocity_factor  # V9: 0.18
                elif combined_dist < 0.08:
                    collision_risk += 0.12 * velocity_factor  # V9: 0.08
                elif combined_dist > 0.15:
                    safe_vehicles += 1
        
        reward -= collision_risk
        
        # Reward maintaining safe distances (keep V8's value)
        reward += safe_vehicles * 0.10
        
        # STRONGER penalty for being surrounded (V9's values weren't enough)
        if very_close >= 3:
            reward -= 0.8  # V9: 0.6
        elif very_close >= 2:
            reward -= 0.5  # V9: 0.35
        
    except (IndexError, TypeError, AttributeError):
        pass
    
    # Modest forward velocity bonus (between V8 and V9)
    try:
        if len(obs) > 0 and len(obs[0]) > 3:
            vx = obs[0][3]
            # Less aggressive than V9, more than V8
            if vx > 0.75:
                reward += 0.22  # V8: 0.20, V9: 0.30
            elif vx > 0.65:
                reward += 0.16  # V8: 0.15, V9: 0.22
            elif vx > 0.55:
                reward += 0.11  # V8: 0.10, V9: 0.14
            elif vx < 0.45:
                reward -= 0.05  # V9: 0.10 (gentler)
    except (IndexError, TypeError, AttributeError):
        pass
    
    # Acceleration control - modest encouragement (between V8 and V9)
    try:
        if hasattr(action, '__len__') and len(action) > 1:
            accel_action = action[1]
            # Penalize extreme actions
            if abs(accel_action) > 0.9:
                reward -= 0.10  # Slightly stronger than V8/V9
            # Gentle bonus for forward acceleration
            elif accel_action > 0.4:
                reward += 0.07  # V8: 0.05, V9: 0.10
            elif accel_action > 0.2:
                reward += 0.04  # V8: similar
    except (IndexError, TypeError, ValueError):
        pass
    
    # Modest compound bonus for optimal safe driving (between V8 and V9)
    if 20.0 <= speed <= 25.0 and on_road and not crashed:
        bonus = 0.18  # V8: 0.15, V9: 0.25
        # Bonus if maintaining safe distances
        if safe_vehicles >= 3:
            bonus += 0.12  # V9: 0.20 (too high)
        elif safe_vehicles >= 2:
            bonus += 0.08  # V9: 0.12 (too high)
        reward += bonus
    
    # Modest speed milestone bonuses (gentler than V9)
    if speed >= 23.0:
        reward += 0.10  # Encourage higher speeds
    elif speed >= 21.0:
        reward += 0.06  # Gradual incentive
    
    return reward
