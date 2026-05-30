# Autonomous Driving Reward Optimization

McNair Research Project — Cleveland State University

## Research Question

Can an LLM-based agent iteratively improve autonomous driving behavior
through agentic reward function optimization?

## Stack

- **Simulator:** highway-env (Gymnasium)
- **RL:** PPO via stable-baselines3
- **Agent:** mini-swe-agent (Claude Sonnet)
- **Logging:** TensorBoard

## Usage

```bash
# Train and evaluate
python train.py

# Watch the agent drive
python evaluate.py

# Run optimization loop (N iterations)
bash optimize_loop.sh 1 10
```

## Structure

- `rewards/baseline.py` — original reward function, never modified
- `rewards/reward_fn.py` — current reward, modified each iteration
- `experiments/` — archived reward versions and metrics per iteration
- `results/` — latest run output
