def compute_reward(obs, action, info):
    """
    Current reward function — modified each iteration by the optimization loop.
    See rewards/baseline.py for the original v0 reference (do not modify that file).
    """
    reward = 0.0
    speed = info.get("speed", 0)
    crashed = info.get("crashed", False)
    on_road = info.get("on_road", True)
    
    # Moderate crash penalty - too high causes training instability
    if crashed:
        reward -= 10.0
        return reward
    
    # Penalty for going off road, but allow recovery
    if not on_road:
        reward -= 2.0
        return reward
    
    # Reward high-speed highway driving with smooth progression
    # Target: 25-30 m/s (90-108 km/h) for highway
    optimal_speed = 25.0
    
    if speed < optimal_speed:
        # Smooth reward progression up to optimal
        reward += (speed / optimal_speed) * 0.8
    else:
        # Reward at optimal speed, slight diminishing returns beyond
        speed_excess = speed - optimal_speed
        if speed_excess <= 10.0:
            reward += 0.8 + speed_excess * 0.02
        else:
            # Cap reward for excessive speed
            reward += 1.0
    
    # Strong survival bonus - reward staying alive
    reward += 0.3
    
    # Bonus for maintaining reasonable speed (not too slow)
    if speed > 10.0:
        reward += 0.15
    
    # Encourage smooth driving with minimal steering
    try:
        if hasattr(action, '__len__') and len(action) > 0:
            steering_action = action[0]
            # Small penalty for steering to encourage stability
            reward -= abs(steering_action) * 0.01
        elif hasattr(action, 'item'):
            steering_action = action.item()
            reward -= abs(steering_action) * 0.01
        else:
            steering_action = float(action)
            reward -= abs(steering_action) * 0.01
    except (IndexError, TypeError, ValueError):
        pass
    
    # Safety reward based on proximity to other vehicles
    try:
        # obs shape is (5, 5): [presence, x, y, vx, vy] for ego + 4 neighbors
        collision_risk = 0.0
        safe_vehicles = 0
        
        for i in range(1, min(5, len(obs))):
            if obs[i][0] > 0.5:  # Vehicle is present
                # x is longitudinal, y is lateral distance (normalized)
                x_dist = abs(obs[i][1])
                y_dist = abs(obs[i][2])
                
                # Calculate combined distance risk
                # Focus more on longitudinal distance for rear-end collisions
                if x_dist < 0.03 and y_dist < 0.15:
                    # Critical danger zone
                    collision_risk += 0.15
                elif x_dist < 0.08 and y_dist < 0.2:
                    # High risk zone
                    collision_risk += 0.08
                elif x_dist > 0.12 or y_dist > 0.3:
                    # Safe distance
                    safe_vehicles += 1
        
        # Penalize high collision risk
        reward -= collision_risk
        
        # Reward maintaining safe distances
        reward += safe_vehicles * 0.03
        
    except (IndexError, TypeError, AttributeError):
        pass
    
    # Bonus for consistent forward motion
    try:
        # Check ego vehicle velocity in x direction (forward)
        if len(obs) > 0 and len(obs[0]) > 3:
            vx = obs[0][3]
            # Reward positive forward velocity
            if vx > 0.5:
                reward += 0.05
    except (IndexError, TypeError, AttributeError):
        pass
    
    return reward
