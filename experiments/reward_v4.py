def compute_reward(obs, action, info):
    """
    Improved reward function (v4) - Eliminate crashes while maintaining speed.
    
    Strategy:
    1. Stronger crash penalty to emphasize zero crashes
    2. Enhanced collision avoidance with time-to-collision awareness
    3. Higher survival bonus to maximize episode completion
    4. Encourage higher speeds safely (targeting 120-130 km/h)
    5. More aggressive penalties for very close proximity
    """
    reward = 0.0
    speed = info.get("speed", 0)
    crashed = info.get("crashed", False)
    on_road = info.get("on_road", True)
    
    # === CRASH PENALTY: Stronger to achieve 0% crash rate ===
    if crashed:
        reward -= 50.0
        return reward
    
    # === OFF-ROAD PENALTY ===
    if not on_road:
        reward -= 10.0
        return reward
    
    # === SURVIVAL BONUS: Higher priority on episode completion ===
    reward += 2.0
    
    # === SPEED REWARD: Encourage faster highway speeds safely ===
    # Target: 32-36 m/s (115-130 km/h) to match v1 performance
    # v1 achieved 138 km/h with 40% crash rate
    # v3 achieved 117 km/h with 13% crash rate
    # Goal: 125-135 km/h with 0% crash rate
    
    if speed < 22.0:
        # Too slow - proportional reward
        reward += speed * 0.10
    elif speed < 30.0:
        # Accelerating - encourage progress
        reward += 2.2 + (speed - 22.0) / 8.0 * 0.8
    elif speed <= 38.0:
        # Optimal highway speed range (108-137 km/h)
        reward += 3.0
    elif speed <= 42.0:
        # Very fast but acceptable
        reward += 3.0 - (speed - 38.0) / 4.0 * 0.6
    else:
        # Excessive speed - moderate penalty
        reward += max(1.8, 2.4 - (speed - 42.0) * 0.12)
    
    # === COLLISION AVOIDANCE: Enhanced with TTC and proximity ===
    try:
        min_forward_dist = float('inf')
        collision_risk_penalty = 0.0
        
        # Analyze nearby vehicles
        for i in range(1, min(5, obs.shape[0])):
            if obs[i, 0] > 0.5:  # Vehicle present
                x_rel = obs[i, 1]  # Longitudinal position
                y_rel = obs[i, 2]  # Lateral position
                vx_rel = obs[i, 3]  # Relative velocity x
                vy_rel = obs[i, 4]  # Relative velocity y
                
                distance = (x_rel ** 2 + y_rel ** 2) ** 0.5
                
                # Time-to-collision estimation
                ttc = float('inf')
                if vx_rel < -0.01:  # Closing in
                    ttc = -x_rel / vx_rel if x_rel > 0 else float('inf')
                
                # === PRIMARY: Vehicles ahead in same lane ===
                if x_rel > 0 and abs(y_rel) < 0.25:
                    min_forward_dist = min(min_forward_dist, distance)
                    
                    # Very aggressive penalties for close proximity
                    if distance < 0.06:
                        collision_risk_penalty += 8.0
                    elif distance < 0.10:
                        collision_risk_penalty += 5.0
                    elif distance < 0.15:
                        collision_risk_penalty += 3.0
                    elif distance < 0.20:
                        collision_risk_penalty += 1.5
                    elif distance < 0.25:
                        collision_risk_penalty += 0.7
                    elif distance < 0.35:
                        collision_risk_penalty += 0.3
                    
                    # Time-to-collision penalty
                    if ttc < 1.0 and distance < 0.4:
                        collision_risk_penalty += 3.0
                    elif ttc < 2.0 and distance < 0.5:
                        collision_risk_penalty += 1.5
                    elif ttc < 3.0 and distance < 0.6:
                        collision_risk_penalty += 0.5
                
                # === SECONDARY: Vehicles in adjacent lanes (side collisions) ===
                elif abs(y_rel) < 0.25:  # Same or adjacent lane
                    if abs(x_rel) < 0.10 and abs(y_rel) < 0.15:
                        # Very close side-by-side
                        collision_risk_penalty += 3.0
                    elif abs(x_rel) < 0.15 and abs(y_rel) < 0.20:
                        collision_risk_penalty += 1.5
                    elif abs(x_rel) < 0.25 and abs(y_rel) < 0.25:
                        collision_risk_penalty += 0.6
                
                # === TERTIARY: Vehicles behind closing fast ===
                if x_rel < 0 and vx_rel < -0.15 and abs(y_rel) < 0.25:
                    # Fast vehicle approaching from behind
                    if distance < 0.15:
                        collision_risk_penalty += 1.0
                    elif distance < 0.25:
                        collision_risk_penalty += 0.4
        
        reward -= collision_risk_penalty
        
        # Bonus for maintaining optimal following distance
        if 0.35 <= min_forward_dist <= 0.7:
            reward += 0.4
        elif 0.25 <= min_forward_dist < 0.35:
            reward += 0.2
        elif 0.7 < min_forward_dist <= 1.0:
            reward += 0.2
        
    except (IndexError, AttributeError, TypeError):
        pass
    
    # === SMOOTH DRIVING: Minimal penalties for necessary maneuvers ===
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
        
        # Only penalize very extreme steering
        if abs(steering) > 0.7:
            reward -= (abs(steering) - 0.7) ** 2 * 0.2
        
        # Minimal penalty for extreme braking/acceleration
        if abs(accel) > 0.85:
            reward -= (abs(accel) - 0.85) * 0.1
        
    except (IndexError, TypeError, ValueError):
        pass
    
    return reward
