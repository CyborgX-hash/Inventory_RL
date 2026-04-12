"""
Benchmark script — Compare heuristic, PPO, and LLM agents across all tasks.

Runs deterministic multi-seed evaluation and outputs a formatted comparison
table with reproducible metrics.

Usage:
    python benchmark.py                    # Heuristic only, 5 seeds
    python benchmark.py --model-dir models # Include PPO comparison
    python benchmark.py --seeds 10         # 10-seed evaluation
    python benchmark.py --output results.json  # Save results to file
"""

import os
import json
import argparse
import numpy as np
from typing import Dict, List

from environment.warehouse_env import WarehouseEnv, load_task_config
from environment.graders import get_grader, SCORE_MIN, SCORE_MAX
from baseline.heuristic_agent import HeuristicAgent


TASK_IDS = [
    "task1_single_product",
    "task2_multi_product",
    "task3_nonstationary",
]


def _safe_score(score: float) -> float:
    """Clamp score to safe open interval."""
    return float(np.clip(score, SCORE_MIN, SCORE_MAX))


def evaluate_heuristic(task_id: str, seeds: List[int]) -> Dict:
    """Run heuristic agent over multiple seeds."""
    config = load_task_config(task_id)
    grader = get_grader(config)

    scores, fill_rates, waste_rates, profits = [], [], [], []

    for seed in seeds:
        env = WarehouseEnv(task_id=task_id, seed=seed)
        agent = HeuristicAgent(env)
        obs, info = env.reset(seed=seed)
        done = False

        while not done:
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)

        score = _safe_score(grader.grade(info))
        profit = info["total_revenue"] - info["total_holding_cost"] - info["total_ordering_cost"]

        scores.append(score)
        fill_rates.append(float(info["fill_rate"]))
        waste_rates.append(float(info["waste_rate"]))
        profits.append(profit)

    return {
        "agent": "heuristic",
        "task_id": task_id,
        "num_seeds": len(seeds),
        "avg_score": round(float(np.mean(scores)), 4),
        "std_score": round(float(np.std(scores)), 4),
        "avg_fill_rate": round(float(np.mean(fill_rates)), 4),
        "avg_waste_rate": round(float(np.mean(waste_rates)), 4),
        "avg_profit": round(float(np.mean(profits)), 2),
        "scores": [round(s, 4) for s in scores],
    }


def evaluate_ppo(task_id: str, model_dir: str, seeds: List[int]) -> Dict:
    """Run PPO agent over multiple seeds."""
    from stable_baselines3 import PPO
    from train import FlattenedWarehouseEnv

    config = load_task_config(task_id)
    grader = get_grader(config)

    best_path = os.path.join(model_dir, task_id, "best_model.zip")
    final_path = os.path.join(model_dir, f"{task_id}_final.zip")

    if os.path.exists(best_path):
        model_path = best_path
    elif os.path.exists(final_path):
        model_path = final_path
    else:
        return {"agent": "ppo", "task_id": task_id, "error": "No model found"}

    model = PPO.load(model_path)
    scores, fill_rates, waste_rates, profits = [], [], [], []

    for seed in seeds:
        env = FlattenedWarehouseEnv(task_id=task_id, seed=seed)
        obs, info = env.reset(seed=seed)
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

        score = _safe_score(grader.grade(info))
        profit = info["total_revenue"] - info["total_holding_cost"] - info["total_ordering_cost"]

        scores.append(score)
        fill_rates.append(float(info["fill_rate"]))
        waste_rates.append(float(info["waste_rate"]))
        profits.append(profit)

    return {
        "agent": "ppo",
        "task_id": task_id,
        "model_path": model_path,
        "num_seeds": len(seeds),
        "avg_score": round(float(np.mean(scores)), 4),
        "std_score": round(float(np.std(scores)), 4),
        "avg_fill_rate": round(float(np.mean(fill_rates)), 4),
        "avg_waste_rate": round(float(np.mean(waste_rates)), 4),
        "avg_profit": round(float(np.mean(profits)), 2),
        "scores": [round(s, 4) for s in scores],
    }


def print_results(all_results: Dict):
    """Print a formatted comparison table."""
    print(f"\n{'='*85}")
    print("BENCHMARK RESULTS")
    print(f"{'='*85}")
    print(
        f"{'Task':<25} {'Agent':<12} {'Score':>10} {'±Std':>8} "
        f"{'Fill Rate':>10} {'Waste':>8} {'Profit':>10}"
    )
    print("-" * 85)

    for task_id in TASK_IDS:
        task_results = all_results.get(task_id, [])
        for r in task_results:
            if "error" in r:
                print(f"{task_id:<25} {r['agent']:<12} {'N/A':>10} {'':>8} {'':>10} {'':>8} {'':>10}")
                continue
            print(
                f"{task_id:<25} {r['agent']:<12} "
                f"{r['avg_score']:>10.4f} {r['std_score']:>8.4f} "
                f"{r['avg_fill_rate']:>10.4f} "
                f"{r['avg_waste_rate']:>8.4f} "
                f"{r['avg_profit']:>10.2f}"
            )
        if task_id != TASK_IDS[-1]:
            print("-" * 85)

    print(f"{'='*85}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark agents on all inventory management tasks"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to trained PPO models. If provided, includes PPO in comparison.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of seeds for evaluation (default: 5)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Starting seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    seeds = list(range(args.base_seed, args.base_seed + args.seeds))

    print(f"\n{'='*60}")
    print("Warehouse Inventory Management — Benchmark")
    print(f"Seeds: {args.seeds} ({seeds[0]}–{seeds[-1]})")
    print(f"Agents: Heuristic" + (", PPO" if args.model_dir else ""))
    print(f"{'='*60}")

    all_results = {}

    for task_id in TASK_IDS:
        print(f"\n  Evaluating {task_id}...")
        task_results = []

        # Heuristic
        print(f"    Heuristic...", end=" ", flush=True)
        result = evaluate_heuristic(task_id, seeds)
        print(f"score={result['avg_score']:.4f}")
        task_results.append(result)

        # PPO
        if args.model_dir:
            print(f"    PPO...", end=" ", flush=True)
            result = evaluate_ppo(task_id, args.model_dir, seeds)
            if "error" in result:
                print(f"({result['error']})")
            else:
                print(f"score={result['avg_score']:.4f}")
            task_results.append(result)

        all_results[task_id] = task_results

    print_results(all_results)

    # Save to file
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
