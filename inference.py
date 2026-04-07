"""
Inference script — Runs trained PPO models (or heuristic baseline) on all tasks
and outputs step-by-step logs with final scores.

Usage:
    python inference.py                      # Run heuristic baseline on all tasks
    python inference.py --model-dir models   # Run trained PPO models
    python inference.py --task task1_single_product  # Single task only
"""

import os
import sys
import json
import argparse
import numpy as np

from environment.warehouse_env import WarehouseEnv, load_task_config
from environment.graders import get_grader
from baseline.heuristic_agent import HeuristicAgent


def run_heuristic(task_id: str, seed: int = 42, verbose: bool = True) -> dict:
    """Run heuristic baseline agent on a task."""
    env = WarehouseEnv(task_id=task_id, seed=seed)
    agent = HeuristicAgent(env)
    config = load_task_config(task_id)
    grader = get_grader(config)

    obs, info = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    step = 0

    if verbose:
        print(json.dumps({"event": "START", "task_id": task_id, "agent": "heuristic", "episode": 1}))

    while not done:
        action = agent.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if verbose:
            step_log = {
                "event": "STEP",
                "step": step,
                "action": action.tolist(),
                "reward": round(float(reward), 4),
                "done": done,
                "fill_rate": round(float(info.get("step_fill_rate", 0)), 4),
                "inventory": env.inventory.tolist(),
            }
            print(json.dumps(step_log))

    score = grader.grade(info)

    end_log = {
        "event": "END",
        "task_id": task_id,
        "agent": "heuristic",
        "total_reward": round(total_reward, 4),
        "score": round(score, 4),
        "fill_rate": round(float(info.get("fill_rate", 0)), 4),
        "waste_rate": round(float(info.get("waste_rate", 0)), 4),
        "total_revenue": round(float(info.get("total_revenue", 0)), 2),
        "total_holding_cost": round(float(info.get("total_holding_cost", 0)), 2),
        "total_ordering_cost": round(float(info.get("total_ordering_cost", 0)), 2),
    }

    if verbose:
        print(json.dumps(end_log))

    return end_log


def run_ppo(task_id: str, model_dir: str, seed: int = 42, verbose: bool = True) -> dict:
    """Run trained PPO model on a task."""
    from stable_baselines3 import PPO
    from train import FlattenedWarehouseEnv

    config = load_task_config(task_id)
    grader = get_grader(config)

    # Try loading best model first, then final
    best_path = os.path.join(model_dir, task_id, "best_model.zip")
    final_path = os.path.join(model_dir, f"{task_id}_final.zip")

    if os.path.exists(best_path):
        model_path = best_path
    elif os.path.exists(final_path):
        model_path = final_path
    else:
        print(f"⚠️  No trained model found for {task_id}, falling back to heuristic")
        return run_heuristic(task_id, seed, verbose)

    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    env = FlattenedWarehouseEnv(task_id=task_id, seed=seed)
    obs, info = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    step = 0

    if verbose:
        print(json.dumps({"event": "START", "task_id": task_id, "agent": "ppo", "episode": 1}))

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if verbose:
            step_log = {
                "event": "STEP",
                "step": step,
                "action": action.tolist() if hasattr(action, 'tolist') else list(action),
                "reward": round(float(reward), 4),
                "done": done,
                "fill_rate": round(float(info.get("step_fill_rate", 0)), 4),
            }
            print(json.dumps(step_log))

    score = grader.grade(info)

    end_log = {
        "event": "END",
        "task_id": task_id,
        "agent": "ppo",
        "total_reward": round(total_reward, 4),
        "score": round(score, 4),
        "fill_rate": round(float(info.get("fill_rate", 0)), 4),
        "waste_rate": round(float(info.get("waste_rate", 0)), 4),
        "total_revenue": round(float(info.get("total_revenue", 0)), 2),
        "total_holding_cost": round(float(info.get("total_holding_cost", 0)), 2),
        "total_ordering_cost": round(float(info.get("total_ordering_cost", 0)), 2),
    }

    if verbose:
        print(json.dumps(end_log))

    return end_log


def main():
    parser = argparse.ArgumentParser(description="Run inference on inventory management tasks")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["task1_single_product", "task2_multi_product", "task3_nonstationary", "all"],
        help="Task to run (default: all)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to trained model directory. If not provided, uses heuristic baseline.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Only print final results, no step logs"
    )
    args = parser.parse_args()

    tasks = (
        ["task1_single_product", "task2_multi_product", "task3_nonstationary"]
        if args.task == "all"
        else [args.task]
    )

    print(f"\n{'='*60}")
    print("Warehouse Inventory Management — Inference")
    print(f"Agent: {'PPO' if args.model_dir else 'Heuristic Baseline'}")
    print(f"{'='*60}\n")

    results = {}
    for task_id in tasks:
        print(f"\n--- {task_id} ---")
        if args.model_dir:
            result = run_ppo(task_id, args.model_dir, args.seed, not args.quiet)
        else:
            result = run_heuristic(task_id, args.seed, not args.quiet)
        results[task_id] = result

    # Summary table
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Task':<25} {'Score':>8} {'Fill Rate':>10} {'Reward':>10}")
    print("-" * 55)
    for tid, r in results.items():
        print(
            f"{tid:<25} {r['score']:>8.4f} {r['fill_rate']:>10.4f} {r['total_reward']:>10.4f}"
        )
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()