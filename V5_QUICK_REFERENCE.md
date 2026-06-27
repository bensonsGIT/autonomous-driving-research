# Reward Function v5 - Quick Reference

## Design Priorities (in order)
1. **Safety First**: Eliminate all crashes (60.0 penalty)
2. **Episode Completion**: Maximize survival (2.5 bonus per step)
3. **High Speed**: Target 125-135 km/h (peak reward at 36-40 m/s)
4. **Collision Avoidance**: Graduated penalties based on proximity and TTC
5. **Smooth Driving**: Minimal action penalties (allow necessary maneuvers)

## Reward Components

| Component | Value | Condition |
|-----------|-------|-----------|
| **Crash Penalty** | -60.0 | Crashed |
| **Off-Road Penalty** | -12.0 | Not on road |
| **Survival Bonus** | +2.5 | Per step alive |
| **Speed Reward** | +4.0 (max) | 36-40 m/s (130-144 km/h) |
| **Following Distance Bonus** | +0.6 | 0.4-0.75 distance (safe) |
| **Close Proximity Penalty** | -12.0 | Distance < 0.05 |
| **TTC Penalty** | -5.0 | TTC < 0.8s, distance < 0.35 |

## Key Improvements from v4

1. **↑ 20% stronger crash penalty** (50 → 60)
2. **↑ 25% higher survival bonus** (2.0 → 2.5)
3. **↑ 33% higher peak speed reward** (3.0 → 4.0)
4. **↑ 50% stronger proximity penalties** (8.0 → 12.0 at closest)
5. **↑ More granular TTC analysis** (additional threshold at 1.5s)
6. **New: Dangerous proximity flag** (prevents unsafe bonuses)

## Expected Improvements

| Metric | v4 | v5 Target | Change |
|--------|-----|-----------|--------|
| Crash Rate | 3.0% | 0-1% | -2-3% |
| Avg Speed | 122.55 km/h | 126-132 km/h | +3-9 km/h |
| Mean Length | 38.88 | 39.5-40.0 | +0.6-1.1 |

## Risk Mitigation

- **If crashes increase**: v5 has stronger penalties than v4
- **If speed drops**: v5 has higher speed rewards than v4  
- **If episodes end early**: v5 has higher survival bonus than v4

The design is conservative and builds incrementally on proven v4 success.
