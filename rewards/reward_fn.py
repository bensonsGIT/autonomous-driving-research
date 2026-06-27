def compute_reward(obs, action, info):
    """
    Improved reward function (v5) - Target: 0% crashes, 125-135 km/h, full episodes.
    
    Strategy:
    1. Very strong crash penalty to eliminate last 3% crashes
    2. Fine-tuned collision avoidance with stricter proximity penalties
    3. Increased speed incentive targeting 125-135 km/h range
    4. Maintain high survival bonus for episode completion
    5. Smarter TTC (time-to-collision) analysis
    """
    reward = 0.0
    speed = info.get("speed", 0)
    crashed = info.get("crashed", False)
    on_road = info.get("on_road", True)
    
    # === CRASH PENALTY: Very strong to achieve 0% crash rate ===
    if crashed:
        reward -= 60.0
        return reward
    
    # === OFF-ROAD PENALTY ===
    if not on_road:
        reward -= 12.0
        return reward
    
    # === SURVIVAL BONUS: Critical for episode completion ===
    reward += 2.5
    
    # === SPEED REWARD: Push higher while maintaining safety ===
    # Target: 33-38 m/s (118-137 km/h)
    # Current v4: 122.55 km/h average
    # Goal v5: 125-135 km/h average
    
    if speed < 20.0:
        # Too slow - proportional reward
        reward += speed * 0.12
    elif speed < 28.0:
        # Accelerating - encourage progress
        reward += 2.4 + (speed - 20.0) / 8.0 * 1.0
    elif speed <= 36.0:
        # Optimal highway speed range (108-130 km/h)
        # Linear increase to encourage higher speeds in this range
        reward += 3.4 + (speed - 28.0) / 8.0 * 0.6
    elif speed <= 40.0:
        # Very fast - maximum reward
        reward += 4.0
    elif speed <= 44.0:
        # Extremely fast but acceptable
        reward += 4.0 - (speed - 40.0) / 4.0 * 0.4
    else:
        # Too fast - moderate penalty
        reward += max(2.2, 3.6 - (speed - 44.0) * 0.15)
    
    # === COLLISION AVOIDANCE: Enhanced precision ===
    try:
        min_forward_dist = float('inf')
        collision_risk_penalty = 0.0
        dangerous_proximity = False
        
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
                
                # === PRIMARY: Vehicles ahead in same/adjacent lane ===
                if x_rel > 0 and abs(y_rel) < 0.3:
                    min_forward_dist = min(min_forward_dist, distance)
                    
                    # Extremely aggressive penalties for very close proximity
                    if distance < 0.05:
                        collision_risk_penalty += 12.0
                        dangerous_proximity = True
                    elif distance < 0.08:
                        collision_risk_penalty += 8.0
                        dangerous_proximity = True
                    elif distance < 0.12:
                        collision_risk_penalty += 5.0
                    elif distance < 0.16:
                        collision_risk_penalty += 3.0
                    elif distance < 0.22:
                        collision_risk_penalty += 1.5
                    elif distance < 0.30:
                        collision_risk_penalty += 0.6
                    elif distance < 0.40:
                        collision_risk_penalty += 0.2
                    
                    # Enhanced TTC penalty - stricter thresholds
                    if ttc < 0.8 and distance < 0.35:
                        collision_risk_penalty += 5.0
                        dangerous_proximity = True
                    elif ttc < 1.5 and distance < 0.45:
                        collision_risk_penalty += 2.5
                    elif ttc < 2.5 and distance < 0.55:
                        collision_risk_penalty += 1.0
                    elif ttc < 3.5 and distance < 0.65:
                        collision_risk_penalty += 0.3
                
                # === SECONDARY: Side collision risk (lane changes) ===
                elif abs(y_rel) < 0.28:
                    if abs(x_rel) < 0.08 and abs(y_rel) < 0.12:
                        # Very close side-by-side - extremely dangerous
                        collision_risk_penalty += 5.0
                        dangerous_proximity = True
                    elif abs(x_rel) < 0.12 and abs(y_rel) < 0.18:
                        collision_risk_penalty += 3.0
                    elif abs(x_rel) < 0.18 and abs(y_rel) < 0.22:
                        collision_risk_penalty += 1.5
                    elif abs(x_rel) < 0.28 and abs(y_rel) < 0.28:
                        collision_risk_penalty += 0.5
                
                # === TERTIARY: Vehicles behind closing very fast ===
                if x_rel < 0 and vx_rel < -0.18 and abs(y_rel) < 0.25:
                    if distance < 0.12:
                        collision_risk_penalty += 2.0
                    elif distance < 0.20:
                        collision_risk_penalty += 0.8
                    elif distance < 0.30:
                        collision_risk_penalty += 0.3
        
        reward -= collision_risk_penalty
        
        # Bonus for maintaining optimal following distance
        # Encourage safe spacing while allowing for speed
        if not dangerous_proximity:
            if 0.4 <= min_forward_dist <= 0.75:
                reward += 0.6
            elif 0.3 <= min_forward_dist < 0.4:
                reward += 0.3
            elif 0.75 < min_forward_dist <= 1.2:
                reward += 0.3
        
    except (IndexError, AttributeError, TypeError):
        pass
    
    # === SMOOTH DRIVING: Minimal interference with necessary maneuvers ===
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
        
        # Penalize steering from a low threshold — smooth highway driving rarely needs >5-10°
        if abs(steering) > 0.1:
            reward -= (abs(steering) - 0.1) ** 2 * 1.5

        # Penalize harsh braking/acceleration
        if abs(accel) > 0.5:
            reward -= (abs(accel) - 0.5) ** 2 * 0.5
        
    except (IndexError, TypeError, ValueError):
        pass
    
    return reward
