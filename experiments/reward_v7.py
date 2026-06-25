def compute_reward(obs, action, info):
    """
    V7 reward function - Focus on safe speed increase.
    V5: 0% crashes, 18.59 m/s (too conservative)
    V6: 29% crashes, 18.53 m/s (failed - relaxed safety too much, didn't increase speed)
    V7: Target 22-25 m/s with <5% crash rate via better speed incentives while keeping V5 safety structure
    """
    reward = 0.0
    speed = info.get("speed", 0)
    crashed = info.get("crashed", False)
    on_road = info.get("on_road", True)
    
    # Restore V5's proven crash penalty - V6's relaxation led to 29% crashes
    if crashed:
        reward -= 15.0
        return reward
    
    # Maintain strong off-road penalty from V5
    if not on_road:
        reward -= 4.0
        return reward
    
    # NEW V7 Speed reward: Stronger exponential incentive for higher speeds
    # Target: 22-26 m/s optimal range
    # Key insight: Make speed increase MORE rewarding, not just less penalized
    optimal_speed_low = 20.0
    optimal_speed_high = 26.0
    danger_speed = 32.0
    
    if speed < optimal_speed_low:
        # Exponential reward curve: speed^1.5 to encourage acceleration
        # At 18.59 m/s: ~0.803, at 20 m/s: ~0.894
        reward += (speed / optimal_speed_low) ** 1.5 * 1.0
    elif speed <= optimal_speed_high:
        # Full reward in optimal range with extra bonus
        # Base reward + bonus for being in sweet spot
        reward += 1.0
        # Progressive bonus within optimal range - highest at 24 m/s
        mid_optimal = (optimal_speed_low + optimal_speed_high) / 2.0
        distance_from_mid = abs(speed - mid_optimal)
        normalized_distance = distance_from_mid / (optimal_speed_high - optimal_speed_low)
        reward += 0.3 * (1.0 - normalized_distance)
    elif speed <= danger_speed:
        # Gentle linear decay beyond optimal
        excess = speed - optimal_speed_high
        penalty_factor = excess / (danger_speed - optimal_speed_high)
        reward += 1.0 * (1.0 - penalty_factor * 0.5)
    else:
        # Strong quadratic penalty for dangerous speeds
        excess = speed - danger_speed
        reward -= (excess * excess) * 0.04
    
    # V5's survival bonus worked well
    reward += 0.4
    
    # NEW: Speed progression bonus - reward moving faster than baseline
    # Specifically designed to encourage 20+ m/s
    if speed >= 24.0:
        reward += 0.35
    elif speed >= 22.0:
        reward += 0.25
    elif speed >= 20.0:
        reward += 0.15
    elif speed >= 18.0:
        # Small bonus for current speed to maintain learning stability
        reward += 0.05
    
    # Restore V5's stricter steering penalties - V6's relaxation contributed to crashes
    try:
        if hasattr(action, '__len__') and len(action) > 0:
            steering_action = action[0]
            # V5's 0.08 coefficient
            reward -= abs(steering_action) * abs(steering_action) * 0.08
        elif hasattr(action, 'item'):
            steering_action = action.item()
            reward -= abs(steering_action) * abs(steering_action) * 0.08
        else:
            steering_action = float(action)
            reward -= abs(steering_action) * abs(steering_action) * 0.08
    except (IndexError, TypeError, ValueError):
        pass
    
    # Restore V5's proven proximity-based safety thresholds
    # V6's relaxation led to 29% crashes
    try:
        collision_risk = 0.0
        safe_vehicles = 0
        very_close = 0
        
        for i in range(1, min(5, len(obs))):
            if obs[i][0] > 0.5:  # Vehicle is present
                x_dist = abs(obs[i][1])
                y_dist = abs(obs[i][2])
                
                # Combined distance metric from V5
                combined_dist = (x_dist * x_dist) + (y_dist * y_dist * 0.5)
                
                # V5's proven thresholds
                if combined_dist < 0.01:
                    # Imminent collision
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
                    # Medium risk zone
                    collision_risk += 0.05
                elif combined_dist > 0.15:
                    # Safe distance maintained - increased reward to encourage speed with safety
                    safe_vehicles += 1
        
        reward -= collision_risk
        
        # Enhanced reward for maintaining safe distances at higher speeds
        reward += safe_vehicles * 0.12
        
        # V5's strong penalties for being surrounded
        if very_close >= 3:
            reward -= 0.5
        elif very_close >= 2:
            reward -= 0.3
        
    except (IndexError, TypeError, AttributeError):
        pass
    
    # Enhanced forward velocity rewards to encourage speed
    try:
        if len(obs) > 0 and len(obs[0]) > 3:
            vx = obs[0][3]
            # Stronger rewards for higher forward velocity
            if vx > 0.75:
                reward += 0.25
            elif vx > 0.65:
                reward += 0.18
            elif vx > 0.55:
                reward += 0.12
            elif vx > 0.45:
                reward += 0.06
    except (IndexError, TypeError, AttributeError):
        pass
    
    # Acceleration control from V5
    try:
        if hasattr(action, '__len__') and len(action) > 1:
            accel_action = action[1]
            # Penalize extreme braking/acceleration
            if abs(accel_action) > 0.85:
                reward -= 0.05
            # Bonus for forward acceleration (not braking)
            elif accel_action > 0.4:
                reward += 0.03
    except (IndexError, TypeError, ValueError):
        pass
    
    # NEW: Compound bonus for optimal performance
    # Reward sustained high-speed safe driving
    if speed >= 22.0 and speed <= 28.0 and on_road and not crashed:
        # Scale bonus based on how many vehicles are at safe distance
        base_bonus = 0.2
        if safe_vehicles >= 3:
            base_bonus += 0.1
        elif safe_vehicles >= 2:
            base_bonus += 0.05
        reward += base_bonus
    
    # NEW: Speed consistency reward over time
    # Penalize very slow speeds more strongly to avoid conservative convergence
    if speed < 15.0:
        reward -= 0.2
    elif speed < 17.0:
        reward -= 0.1
    
    return reward
