"""
Averages per-seed fixed_metrics files into a single results/fixed_metrics.json.
"""
import argparse
import json
import os
import numpy as np

OUTPUT_PATH = "results/fixed_metrics.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 0])
    args = parser.parse_args()

    all_metrics = []
    for seed in args.seeds:
        path = f"results/fixed_metrics_seed_{seed}.json"
        if not os.path.exists(path):
            print(f"[avg_metrics] Missing {path}, skipping seed {seed}.")
            continue
        with open(path) as f:
            all_metrics.append(json.load(f))

    if not all_metrics:
        print("[avg_metrics] No seed metrics found.")
        raise SystemExit(1)

    averaged = {
        "seeds": [m["seed"] for m in all_metrics],
        "episodes_per_seed": all_metrics[0]["episodes"],
        "crash_rate": round(float(np.mean([m["crash_rate"] for m in all_metrics])), 4),
        "crash_rate_std": round(float(np.std([m["crash_rate"] for m in all_metrics])), 4),
        "total_crashes": round(float(np.mean([m["total_crashes"] for m in all_metrics])), 2),
        "mean_length": round(float(np.mean([m["mean_length"] for m in all_metrics])), 2),
        "mean_length_std": round(float(np.std([m["mean_length"] for m in all_metrics])), 2),
        "avg_speed_ms": round(float(np.mean([m["avg_speed_ms"] for m in all_metrics])), 2),
        "avg_speed_kmh": round(float(np.mean([m["avg_speed_kmh"] for m in all_metrics])), 2),
        "avg_speed_kmh_std": round(float(np.std([m["avg_speed_kmh"] for m in all_metrics])), 2),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(averaged, f, indent=2)

    print(f"[avg_metrics] crash_rate={averaged['crash_rate']:.2%} ± {averaged['crash_rate_std']:.2%}  "
          f"avg_speed={averaged['avg_speed_kmh']} ± {averaged['avg_speed_kmh_std']} km/h  "
          f"mean_length={averaged['mean_length']} ± {averaged['mean_length_std']}")


if __name__ == "__main__":
    main()
