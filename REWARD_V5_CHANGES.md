# Reward Function v5 Changes

## Objective
Improve upon v4's excellent performance (3% crash rate, 122.55 km/h, 38.88 episode length) to achieve:
- Target crash rate: 0%
- Target speed: 125-135 km/h
- Target episode length: 40

## Key Changes from v4

### 1. Increased Crash Penalty (50 → 60)
- **Rationale**: Eliminate the last 3% crashes (3 out of 300 episodes)
- **Impact**: Stronger deterrent for risky behaviors

### 2. Increased Off-Road Penalty (10 → 12)
- **Rationale**: Ensure agent stays on road in all scenarios
- **Impact**: More consistent lane discipline

### 3. Higher Survival Bonus (2.0 → 2.5)
- **Rationale**: Maximize episode completion rate
- **Impact**: Stronger incentive to survive full 40 steps

### 4. Enhanced Speed Reward Structure
- **Change**: Restructured speed rewards to encourage 33-38 m/s (118-137 km/h)
- **New peak reward**: 4.0 for speeds 36-40 m/s (vs 3.0 in v4)
- **Rationale**: Push average speed from 122.55 to 125-135 km/h range
- **Impact**: Higher rewards for sustained high speeds

### 5. Stricter Collision Avoidance
- **Enhanced proximity penalties**:
  - Distance < 0.05: 12.0 penalty (vs 8.0 in v4)
  - Distance < 0.08: 8.0 penalty (vs 5.0 in v4)
- **Improved TTC thresholds**:
  - TTC < 0.8s + distance < 0.35: 5.0 penalty (stricter)
  - TTC < 1.5s + distance < 0.45: 2.5 penalty (new tier)
- **Dangerous proximity flag**: Prevents following distance bonus when in dangerous situations
- **Rationale**: Prevent the 3 crashes that still occurred in v4
- **Impact**: Agent will maintain safer distances at high speeds

### 6. Improved Following Distance Bonus
- **New optimal range**: 0.4-0.75 distance (vs 0.35-0.7 in v4)
- **Increased bonus**: 0.6 (vs 0.4 in v4)
- **Safety check**: Only awarded when not in dangerous proximity
- **Rationale**: Encourage safe spacing at higher speeds
- **Impact**: More consistent safe following behavior

### 7. Slightly Relaxed Action Penalties
- **Steering threshold**: 0.75 (vs 0.7 in v4)
- **Acceleration threshold**: 0.9 (vs 0.85 in v4)
- **Rationale**: Allow necessary evasive maneuvers to avoid crashes
- **Impact**: Better crash avoidance capability

## Expected Outcomes

Based on v4's strong performance:
- **Crash rate**: 3% → 0-1% (eliminate remaining crashes)
- **Average speed**: 122.55 km/h → 126-132 km/h (+3-9 km/h)
- **Episode length**: 38.88 → 39.5-40.0 (near-perfect completion)

## Design Philosophy

The v5 reward function maintains v4's successful conservative approach while making targeted improvements:
1. Stronger safety signals (crash/proximity penalties)
2. Higher speed incentives within safe operating envelope
3. Better balance between speed and safety
4. Smarter collision avoidance with TTC analysis
