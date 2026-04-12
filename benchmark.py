"""
Benchmark script — Compare heuristic, fixed-quantity, and PPO agents across all tasks.

Runs deterministic multi-seed evaluation and outputs a formatted comparison
table with reproducible metrics.

Usage:
    python benchmark.py                    # Heuristic + fixed-quantity, 5 seeds
    python benchmark.py --model-dir models # Include PPO comparison
    python benchmark.py --seeds 10         # 10-seed evaluation
    python benchmark.py --output results/benchmark.json  # Save results
    python benchmark.py --format markdown  # Output as markdown table
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np

from baseline.fixed_quantity_agent import FixedQuantityAgent
from baseline.heuristic_agent import HeuristicAgent
from environment.graders import SCORE_MAX, SCORE_MIN, get_grader
from environment.warehouse_env import WarehouseEnv, load_task_config

TASK_IDS = [
    "task1_single_product",
    "task2_multi_product",
    "task3_nonstationary",
]


def _safe_score(score: float) -> float:
    """Clamp score to safe open interval."""
    return float(np.clip(score, SCORE_MIN, SCORE_MAX))


def _run_agent_episode(env, agent, grader, seed):
    """Run a single episode and return metrics dict."""
    obs, info = env.reset(seed=seed)
    done = False

    while not done:
        action = agent.act(obs)
        obs, reward, done, truncated, info = env.step(action)

    score = _safe_score(grader.grade(info))
    profit = info["total_revenue"] - info["total_holding_cost"] - info["total_ordering_cost"]

    return {
        "score": score,
        "fill_rate": float(info["fill_rate"]),
        "waste_rate": float(info["waste_rate"]),
        "profit": profit,
        "service_level": float(info.get("service_level", 0)),
    }


def evaluate_heuristic(task_id: str, seeds: list[int]) -> dict:
    """Run heuristic agent over multiple seeds."""
    config = load_task_config(task_id)
    grader = get_grader(config)

    results = []
    for seed in seeds:
        env = WarehouseEnv(task_id=task_id, seed=seed)
        agent = HeuristicAgent(env)
        results.append(_run_agent_episode(env, agent, grader, seed))

    return _aggregate_results("heuristic", task_id, results, seeds)


def evaluate_fixed_quantity(task_id: str, seeds: list[int]) -> dict:
    """Run fixed-quantity agent over multiple seeds."""
    config = load_task_config(task_id)
    grader = get_grader(config)

    results = []
    for seed in seeds:
        env = WarehouseEnv(task_id=task_id, seed=seed)
        agent = FixedQuantityAgent(env, action_index=2)  # Always order 10 units
        results.append(_run_agent_episode(env, agent, grader, seed))

    return _aggregate_results("fixed_qty", task_id, results, seeds)


def evaluate_ppo(task_id: str, model_dir: str, seeds: list[int]) -> dict:
    """Run PPO agent over multiple seeds."""
    from stable_baselines3 import PPO

    from train import FlattenedWarehouseEnv

    config = load_task_config(task_id)

    best_path = os.path.join(model_dir, task_id, "best_model.zip")
    final_path = os.path.join(model_dir, f"{task_id}_final.zip")

    if os.path.exists(best_path):
        model_path = best_path
    elif os.path.exists(final_path):
        model_path = final_path
    else:
        return {"agent": "ppo", "task_id": task_id, "error": "No model found"}

    model = PPO.load(model_path)

    results = []
    for seed in seeds:
        env = FlattenedWarehouseEnv(task_id=task_id, seed=seed)
        obs, info = env.reset(seed=seed)
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

        score = _safe_score(get_grader(config).grade(info))
        profit = info["total_revenue"] - info["total_holding_cost"] - info["total_ordering_cost"]
        results.append({
            "score": score,
            "fill_rate": float(info["fill_rate"]),
            "waste_rate": float(info["waste_rate"]),
            "profit": profit,
            "service_level": float(info.get("service_level", 0)),
        })

    return _aggregate_results("ppo", task_id, results, seeds, model_path=model_path)


def _aggregate_results(
    agent_name: str,
    task_id: str,
    results: list[dict],
    seeds: list[int],
    model_path: str = None,
) -> dict:
    """Aggregate per-seed results into summary statistics."""
    scores = [r["score"] for r in results]
    fill_rates = [r["fill_rate"] for r in results]
    waste_rates = [r["waste_rate"] for r in results]
    profits = [r["profit"] for r in results]
    service_levels = [r["service_level"] for r in results]

    summary = {
        "agent": agent_name,
        "task_id": task_id,
        "num_seeds": len(seeds),
        "avg_score": round(float(np.mean(scores)), 4),
        "std_score": round(float(np.std(scores)), 4),
        "avg_fill_rate": round(float(np.mean(fill_rates)), 4),
        "avg_waste_rate": round(float(np.mean(waste_rates)), 4),
        "avg_profit": round(float(np.mean(profits)), 2),
        "avg_service_level": round(float(np.mean(service_levels)), 4),
        "scores": [round(s, 4) for s in scores],
    }
    if model_path:
        summary["model_path"] = model_path
    return summary


def print_results(all_results: dict, fmt: str = "table"):
    """Print a formatted comparison table."""
    if fmt == "markdown":
        _print_markdown(all_results)
    else:
        _print_table(all_results)


def _print_table(all_results: dict):
    """Print as aligned ASCII table."""
    print(f"\n{'=' * 95}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 95}")
    print(
        f"{'Task':<25} {'Agent':<12} {'Score':>10} {'±Std':>8} "
        f"{'Fill Rate':>10} {'Waste':>8} {'SvcLvl':>8} {'Profit':>10}"
    )
    print("-" * 95)

    for task_id in TASK_IDS:
        task_results = all_results.get(task_id, [])
        for r in task_results:
            if "error" in r:
                print(f"{task_id:<25} {r['agent']:<12} {'N/A':>10}")
                continue
            print(
                f"{task_id:<25} {r['agent']:<12} "
                f"{r['avg_score']:>10.4f} {r['std_score']:>8.4f} "
                f"{r['avg_fill_rate']:>10.4f} "
                f"{r['avg_waste_rate']:>8.4f} "
                f"{r.get('avg_service_level', 0):>8.4f} "
                f"{r['avg_profit']:>10.2f}"
            )
        if task_id != TASK_IDS[-1]:
            print("-" * 95)

    print(f"{'=' * 95}\n")


def _print_markdown(all_results: dict):
    """Print as markdown table (README-ready)."""
    print("\n| Task | Agent | Score | ±Std | Fill Rate | Waste | Svc Level | Profit |")
    print("|------|-------|:-----:|:----:|:---------:|:-----:|:---------:|:------:|")

    for task_id in TASK_IDS:
        task_results = all_results.get(task_id, [])
        for r in task_results:
            if "error" in r:
                print(f"| {task_id} | {r['agent']} | N/A | — | — | — | — | — |")
                continue
            print(
                f"| {task_id} | {r['agent']} "
                f"| {r['avg_score']:.4f} | {r['std_score']:.4f} "
                f"| {r['avg_fill_rate']:.4f} "
                f"| {r['avg_waste_rate']:.4f} "
                f"| {r.get('avg_service_level', 0):.4f} "
                f"| {r['avg_profit']:.2f} |"
            )

    print()


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
        help="Save results to JSON file (auto-creates directory)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["table", "markdown"],
        default="table",
        help="Output format (default: table)",
    )
    args = parser.parse_args()

    seeds = list(range(args.base_seed, args.base_seed + args.seeds))

    agents = ["Heuristic", "Fixed-Qty"]
    if args.model_dir:
        agents.append("PPO")

    print(f"\n{'=' * 60}")
    print("Warehouse Inventory Management — Benchmark")
    print(f"Seeds: {args.seeds} ({seeds[0]}–{seeds[-1]})")
    print(f"Agents: {', '.join(agents)}")
    print(f"{'=' * 60}")

    all_results = {}

    for task_id in TASK_IDS:
        print(f"\n  Evaluating {task_id}...")
        task_results = []

        # Heuristic
        print("    Heuristic...", end=" ", flush=True)
        result = evaluate_heuristic(task_id, seeds)
        print(f"score={result['avg_score']:.4f}")
        task_results.append(result)

        # Fixed quantity
        print("    Fixed-Qty...", end=" ", flush=True)
        result = evaluate_fixed_quantity(task_id, seeds)
        print(f"score={result['avg_score']:.4f}")
        task_results.append(result)

        # PPO
        if args.model_dir:
            print("    PPO...", end=" ", flush=True)
            result = evaluate_ppo(task_id, args.model_dir, seeds)
            if "error" in result:
                print(f"({result['error']})")
            else:
                print(f"score={result['avg_score']:.4f}")
            task_results.append(result)

        all_results[task_id] = task_results

    print_results(all_results, fmt=args.format)

    # Save to file
    output_path = args.output
    if output_path is None:
        # Auto-save to results/ with timestamp
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/benchmark_{timestamp}.json"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
