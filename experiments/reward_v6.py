def compute_reward(obs, action, info):
    """
    V6 reward function - Optimized for higher safe speeds while maintaining low crash rate.
    V5 achieved 0% crashes but speed was too low (18.59 m/s). Target: 24-26 m/s with <5% crash rate.
    """
    reward = 0.0
    speed = info.get("speed", 0)
    crashed = info.get("crashed", False)
    on_road = info.get("on_road", True)
    
    # Maintain moderate crash penalty that worked well in V5
    if crashed:
        reward -= 15.0
        return reward
    
    # Keep strong off-road penalty
    if not on_road:
        reward -= 4.0
        return reward
    
    # Enhanced speed reward structure - encourage higher speeds in optimal range
    # V5 was too conservative, V6 pushes for 22-28 m/s sweet spot
    optimal_speed_low = 18.0  # Lowered from 20 to not penalize current speed
    optimal_speed_peak_low = 22.0  # Peak reward range start
    optimal_speed_peak_high = 28.0  # Peak reward range end
    danger_speed = 33.0  # Increased from 32 to allow slightly higher speeds
    
    if speed < optimal_speed_low:
        # Linear reward up to optimal range
        reward += (speed / optimal_speed_low) * 0.7
    elif speed < optimal_speed_peak_low:
        # Gradual increase from 0.7 to 1.0 as speed increases to peak
        progress = (speed - optimal_speed_low) / (optimal_speed_peak_low - optimal_speed_low)
        reward += 0.7 + (0.3 * progress)
    elif speed <= optimal_speed_peak_high:
        # Full reward in peak range - maximum incentive for 22-28 m/s
        reward += 1.0
    elif speed <= danger_speed:
        # Gentle linear decay beyond optimal
        excess = speed - optimal_speed_peak_high
        penalty_factor = excess / (danger_speed - optimal_speed_peak_high)
        reward += 1.0 * (1.0 - penalty_factor * 0.6)
    else:
        # Strong quadratic penalty for dangerous speeds
        excess = speed - danger_speed
        reward -= (excess * excess) * 0.05
    
    # Increased survival bonus to reward longer episodes
    reward += 0.5
    
    # Enhanced speed maintenance bonus with better range
    if 20.0 <= speed <= 32.0:
        reward += 0.2
    elif 16.0 <= speed < 20.0:
        # Smaller bonus for current speed range
        reward += 0.1
    
    # Reduced steering penalty to allow more dynamic driving
    try:
        if hasattr(action, '__len__') and len(action) > 0:
            steering_action = action[0]
            # Reduced quadratic penalty coefficient from 0.08 to 0.06
            reward -= abs(steering_action) * abs(steering_action) * 0.06
        elif hasattr(action, 'item'):
            steering_action = action.item()
            reward -= abs(steering_action) * abs(steering_action) * 0.06
        else:
            steering_action = float(action)
            reward -= abs(steering_action) * abs(steering_action) * 0.06
    except (IndexError, TypeError, ValueError):
        pass
    
    # Slightly relaxed proximity-based safety since V5 had 0% crashes
    try:
        collision_risk = 0.0
        safe_vehicles = 0
        very_close = 0
        
        for i in range(1, min(5, len(obs))):
            if obs[i][0] > 0.5:  # Vehicle is present
                x_dist = abs(obs[i][1])
                y_dist = abs(obs[i][2])
                
                # Combined distance metric
                combined_dist = (x_dist * x_dist) + (y_dist * y_dist * 0.5)
                
                if combined_dist < 0.008:
                    # Imminent collision - slightly reduced from 0.01
                    collision_risk += 0.6
                    very_close += 1
                elif combined_dist < 0.02:
                    # Critical danger zone - slightly reduced from 0.025
                    collision_risk += 0.35
                    very_close += 1
                elif combined_dist < 0.045:
                    # High risk zone - slightly reduced from 0.05
                    collision_risk += 0.18
                elif combined_dist < 0.075:
                    # Medium risk zone - slightly reduced from 0.08
                    collision_risk += 0.06
                elif combined_dist > 0.12:
                    # Safe distance maintained - slightly reduced threshold from 0.15
                    safe_vehicles += 1
        
        reward -= collision_risk
        
        # Enhanced reward for maintaining safe distances
        reward += safe_vehicles * 0.1
        
        # Maintain strong penalty for being surrounded
        if very_close >= 3:
            reward -= 0.6
        elif very_close >= 2:
            reward -= 0.35
        
    except (IndexError, TypeError, AttributeError):
        pass
    
    # Enhanced forward velocity rewards
    try:
        if len(obs) > 0 and len(obs[0]) > 3:
            vx = obs[0][3]
            # Stronger rewards for higher forward velocity
            if vx > 0.75:
                reward += 0.2
            elif vx > 0.6:
                reward += 0.12
            elif vx > 0.45:
                reward += 0.06
    except (IndexError, TypeError, AttributeError):
        pass
    
    # Maintain acceleration control bonus
    try:
        if hasattr(action, '__len__') and len(action) > 1:
            accel_action = action[1]
            # Encourage moderate acceleration
            if abs(accel_action) > 0.8:
                reward -= 0.06
            elif 0.3 < abs(accel_action) <= 0.6:
                # Small bonus for active but not extreme acceleration
                reward += 0.02
    except (IndexError, TypeError, ValueError):
        pass
    
    # New: Bonus for maintaining consistent high performance
    if speed >= 22.0 and speed <= 30.0 and on_road and not crashed:
        reward += 0.15
    
    return reward
