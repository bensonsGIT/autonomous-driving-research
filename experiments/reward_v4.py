def compute_reward(obs, action, info):
    """
    Current reward function — modified each iteration by the optimization loop.
    See rewards/baseline.py for the original v0 reference (do not modify that file).
    """
    reward = 0.0
    speed = info.get("speed", 0)
    crashed = info.get("crashed", False)
    on_road = info.get("on_road", True)
    
    # Increased crash penalty to discourage risky behavior (v3: -10 -> v4: -20)
    if crashed:
        reward -= 20.0
        return reward
    
    # Moderate penalty for going off road, still allow recovery
    if not on_road:
        reward -= 3.0
        return reward
    
    # Reward high-speed highway driving with emphasis on safety
    # Target: 25-30 m/s (90-108 km/h) for highway
    # Problem: agent was reaching 38.5 m/s (138 km/h) - too fast!
    optimal_speed = 25.0
    max_safe_speed = 32.0  # ~115 km/h
    
    if speed < optimal_speed:
        # Smooth reward progression up to optimal
        reward += (speed / optimal_speed) * 1.0
    elif speed <= max_safe_speed:
        # Reward at optimal, slight bonus up to max safe
        speed_excess = speed - optimal_speed
        reward += 1.0 + speed_excess * 0.01
    else:
        # Penalty for excessive speed beyond safe limits
        speed_excess = speed - max_safe_speed
        reward += 1.07 - speed_excess * 0.05
    
    # Increased survival bonus - staying alive is critical (v3: 0.3 -> v4: 0.5)
    reward += 0.5
    
    # Bonus for maintaining reasonable speed (not too slow)
    if speed > 12.0:
        reward += 0.2
    
    # Encourage smooth driving with minimal steering
    try:
        if hasattr(action, '__len__') and len(action) > 0:
            steering_action = action[0]
            # Small penalty for steering to encourage stability
            reward -= abs(steering_action) * 0.015
        elif hasattr(action, 'item'):
            steering_action = action.item()
            reward -= abs(steering_action) * 0.015
        else:
            steering_action = float(action)
            reward -= abs(steering_action) * 0.015
    except (IndexError, TypeError, ValueError):
        pass
    
    # Enhanced safety reward based on proximity to other vehicles
    try:
        # obs shape is (5, 5): [presence, x, y, vx, vy] for ego + 4 neighbors
        collision_risk = 0.0
        safe_vehicles = 0
        dangerously_close = 0
        
        for i in range(1, min(5, len(obs))):
            if obs[i][0] > 0.5:  # Vehicle is present
                # x is longitudinal, y is lateral distance (normalized)
                x_dist = abs(obs[i][1])
                y_dist = abs(obs[i][2])
                
                # Calculate combined distance risk with stronger penalties
                # Focus more on longitudinal distance for rear-end collisions
                if x_dist < 0.05 and y_dist < 0.15:
                    # Critical danger zone - imminent collision risk
                    collision_risk += 0.3
                    dangerously_close += 1
                elif x_dist < 0.1 and y_dist < 0.2:
                    # High risk zone - too close
                    collision_risk += 0.15
                    dangerously_close += 1
                elif x_dist < 0.15 and y_dist < 0.25:
                    # Medium risk zone - getting close
                    collision_risk += 0.05
                elif x_dist > 0.2 or y_dist > 0.35:
                    # Safe distance maintained
                    safe_vehicles += 1
        
        # Strong penalty for high collision risk
        reward -= collision_risk
        
        # Reward maintaining safe distances
        reward += safe_vehicles * 0.05
        
        # Additional penalty for multiple vehicles too close
        if dangerously_close >= 2:
            reward -= 0.2
        
    except (IndexError, TypeError, AttributeError):
        pass
    
    # Bonus for consistent forward motion
    try:
        # Check ego vehicle velocity in x direction (forward)
        if len(obs) > 0 and len(obs[0]) > 3:
            vx = obs[0][3]
            # Reward positive forward velocity
            if vx > 0.6:
                reward += 0.1
    except (IndexError, TypeError, AttributeError):
        pass
    
    return reward
