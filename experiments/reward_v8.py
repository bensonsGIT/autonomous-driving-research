def compute_reward(obs, action, info):
    """
    V8 reward function - Refined approach for safe high-speed driving.
    V7: 24% crashes, 19.59 m/s (improved speed but crash rate too high)
    V8: Target 21-24 m/s with <10% crash rate via smarter safety + cleaner speed incentives
    
    Key changes:
    - Stronger crash penalty to reinforce safety
    - Simplified speed reward curve (fewer bonuses, clearer structure)
    - Relative velocity awareness in collision detection (safer close-following at matched speeds)
    - Better balance between speed and safety rewards
    """
    reward = 0.0
    speed = info.get("speed", 0)
    crashed = info.get("crashed", False)
    on_road = info.get("on_road", True)
    
    # Stronger crash penalty - V7's -15 wasn't enough deterrent
    if crashed:
        reward -= 20.0
        return reward
    
    # Strong off-road penalty
    if not on_road:
        reward -= 4.0
        return reward
    
    # Simplified speed reward curve - clearer signal for learning
    # Target: 20-26 m/s optimal range
    optimal_speed_low = 20.0
    optimal_speed_high = 26.0
    danger_speed = 32.0
    
    if speed < optimal_speed_low:
        # Smoother exponential curve to encourage reaching optimal range
        # At 15 m/s: 0.65, at 18 m/s: 0.82, at 20 m/s: 1.0
        reward += (speed / optimal_speed_low) ** 1.3
    elif speed <= optimal_speed_high:
        # Constant high reward in optimal range (no complex bonuses)
        reward += 1.0
        # Small bonus at the sweet spot (23 m/s)
        if 22.0 <= speed <= 24.0:
            reward += 0.15
    elif speed <= danger_speed:
        # Gradual decay beyond optimal
        excess = speed - optimal_speed_high
        penalty_factor = excess / (danger_speed - optimal_speed_high)
        reward += 1.0 * (1.0 - penalty_factor * 0.7)
    else:
        # Quadratic penalty for dangerous speeds
        excess = speed - danger_speed
        reward -= (excess ** 2) * 0.05
    
    # Base survival bonus
    reward += 0.35
    
    # Penalize very slow speeds to prevent over-conservative behavior
    if speed < 15.0:
        reward -= 0.25
    elif speed < 17.0:
        reward -= 0.12
    
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
    
    # Enhanced proximity-based safety with relative velocity awareness
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
                
                # Velocity-adjusted risk: closer is safer if velocities match
                # If relative velocity is low, reduce risk penalty
                velocity_factor = min(1.0, 0.3 + rel_v_total * 2.0)
                
                if combined_dist < 0.008:
                    # Imminent collision
                    collision_risk += 0.6 * velocity_factor
                    very_close += 1
                elif combined_dist < 0.02:
                    # Critical danger zone
                    collision_risk += 0.35 * velocity_factor
                    very_close += 1
                elif combined_dist < 0.045:
                    # High risk zone
                    collision_risk += 0.18 * velocity_factor
                elif combined_dist < 0.08:
                    # Medium risk zone
                    collision_risk += 0.08 * velocity_factor
                elif combined_dist > 0.15:
                    # Safe distance maintained
                    safe_vehicles += 1
        
        reward -= collision_risk
        
        # Reward maintaining safe distances (but don't overdo it)
        reward += safe_vehicles * 0.10
        
        # Strong penalty for being surrounded
        if very_close >= 3:
            reward -= 0.6
        elif very_close >= 2:
            reward -= 0.35
        
    except (IndexError, TypeError, AttributeError):
        pass
    
    # Forward velocity bonus (aligned with speed goals)
    try:
        if len(obs) > 0 and len(obs[0]) > 3:
            vx = obs[0][3]
            if vx > 0.7:
                reward += 0.20
            elif vx > 0.6:
                reward += 0.15
            elif vx > 0.5:
                reward += 0.10
    except (IndexError, TypeError, AttributeError):
        pass
    
    # Acceleration control - encourage forward acceleration, penalize jerky behavior
    try:
        if hasattr(action, '__len__') and len(action) > 1:
            accel_action = action[1]
            # Penalize extreme actions
            if abs(accel_action) > 0.9:
                reward -= 0.08
            # Bonus for moderate forward acceleration
            elif 0.3 < accel_action < 0.7:
                reward += 0.05
    except (IndexError, TypeError, ValueError):
        pass
    
    # Compound bonus for optimal high-speed safe driving
    # This is the behavior we want to reinforce most strongly
    if 21.0 <= speed <= 27.0 and on_road and not crashed:
        bonus = 0.15
        # Extra bonus if maintaining safe distances
        if safe_vehicles >= 2:
            bonus += 0.10
        reward += bonus
    
    return reward
