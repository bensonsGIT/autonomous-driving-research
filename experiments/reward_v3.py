def compute_reward(obs, action, info):
    """
    Improved reward function (v3) - Focus on survival first, then speed.
    
    Strategy:
    1. Moderate crash penalty for better learning gradient
    2. Strong survival bonus to maximize episode length
    3. Simple collision avoidance focused on forward distance
    4. Encourage highway speeds but not at cost of safety
    5. Minimal action penalties to allow necessary maneuvers
    """
    reward = 0.0
    speed = info.get("speed", 0)
    crashed = info.get("crashed", False)
    on_road = info.get("on_road", True)
    
    # === CRASH PENALTY: Moderate for better learning ===
    if crashed:
        reward -= 40.0
        return reward
    
    # === OFF-ROAD PENALTY ===
    if not on_road:
        reward -= 8.0
        return reward
    
    # === SURVIVAL BONUS: Primary objective ===
    # Every step alive is valuable
    reward += 1.5
    
    # === SPEED REWARD: Encourage progression toward highway speeds ===
    # Target: 30-35 m/s (108-126 km/h)
    # The v1 baseline achieved 138 km/h, so we need to encourage higher speeds
    
    if speed < 20.0:
        # Too slow - proportional reward to encourage acceleration
        reward += speed * 0.08
    elif speed < 28.0:
        # Accelerating - good progress
        reward += 1.6 + (speed - 20.0) / 8.0 * 1.0
    elif speed <= 35.0:
        # Optimal highway speed range - maximum reward
        reward += 2.6
    elif speed <= 40.0:
        # Fast but still acceptable
        reward += 2.6 - (speed - 35.0) / 5.0 * 0.4
    else:
        # Very fast - risky but don't penalize too much
        reward += max(1.5, 2.2 - (speed - 40.0) * 0.15)
    
    # === COLLISION AVOIDANCE: Simple forward distance tracking ===
    try:
        min_forward_dist = float('inf')
        collision_risk_penalty = 0.0
        
        # Analyze nearby vehicles
        for i in range(1, min(5, obs.shape[0])):
            if obs[i, 0] > 0.5:  # Vehicle present
                x_rel = obs[i, 1]  # Longitudinal position
                y_rel = obs[i, 2]  # Lateral position
                vx_rel = obs[i, 3]  # Relative velocity
                
                distance = (x_rel ** 2 + y_rel ** 2) ** 0.5
                
                # Primary concern: vehicles directly ahead
                if x_rel > 0 and abs(y_rel) < 0.25:
                    min_forward_dist = min(min_forward_dist, distance)
                    
                    # Graduated penalties based on proximity
                    if distance < 0.08:
                        collision_risk_penalty += 4.0
                    elif distance < 0.12:
                        collision_risk_penalty += 2.0
                    elif distance < 0.18:
                        collision_risk_penalty += 1.0
                    elif distance < 0.25:
                        collision_risk_penalty += 0.4
                    
                    # Additional penalty if rapidly closing
                    if vx_rel < -0.1 and distance < 0.3:
                        collision_risk_penalty += 0.8
                
                # Side collision risk (lane changes)
                elif abs(x_rel) < 0.12 and abs(y_rel) < 0.12:
                    collision_risk_penalty += 1.5
                elif abs(x_rel) < 0.2 and abs(y_rel) < 0.2:
                    collision_risk_penalty += 0.5
        
        reward -= collision_risk_penalty
        
        # Bonus for maintaining good following distance
        if 0.3 <= min_forward_dist <= 0.6:
            reward += 0.3
        elif 0.6 < min_forward_dist <= 1.0:
            reward += 0.15
        
    except (IndexError, AttributeError, TypeError):
        pass
    
    # === SMOOTH DRIVING: Only penalize extreme actions ===
    try:
        if hasattr(action, '__len__') and len(action) > 0:
            steering = action[0]
            accel = action[1] if len(action) > 1 else 0.0
        elif hasattr(action, 'item'):
            steering = action.item()
            accel = 0.0
        else:
            steering = float(action)
            accel = 0.0
        
        # Light penalty only for very aggressive steering
        if abs(steering) > 0.6:
            reward -= (abs(steering) - 0.6) ** 2 * 0.15
        
        # Minimal penalty for extreme acceleration changes
        if abs(accel) > 0.8:
            reward -= (abs(accel) - 0.8) * 0.08
        
    except (IndexError, TypeError, ValueError):
        pass
    
    return reward
