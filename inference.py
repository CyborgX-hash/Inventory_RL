"""
Inference script — Runs trained PPO models, heuristic baseline, or LLM agent
on all tasks and outputs step-by-step logs with final scores.

Usage:
    python inference.py                      # Run heuristic baseline on all tasks
    python inference.py --model-dir models   # Run trained PPO models
    python inference.py --use-llm            # Run LLM agent via OpenAI API
    python inference.py --task task1_single_product  # Single task only
    python inference.py --num-seeds 5        # Multi-seed evaluation
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Optional, List

# Hackathon Checklist Requirements
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo-preview")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

from environment.warehouse_env import WarehouseEnv, load_task_config, ORDER_LEVELS, EMERGENCY_ORDER_LEVELS
from environment.graders import get_grader, SCORE_MIN, SCORE_MAX
from baseline.heuristic_agent import HeuristicAgent


def _safe_score(score: float) -> float:
    """Clamp score to safe open interval."""
    return float(np.clip(score, SCORE_MIN, SCORE_MAX))


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

    print(f"[START] task={task_id}", flush=True)
    if verbose:
        print(json.dumps({"event": "START", "task_id": task_id, "agent": "heuristic", "episode": 1}), flush=True)

    while not done:
        action = agent.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        print(f"[STEP] step={step} reward={float(reward):.4f}", flush=True)
        if verbose:
            step_log = {
                "event": "STEP",
                "step": step,
                "action_ids": action.tolist(),
                "reward": round(float(reward), 4),
                "done": done,
                "fill_rate": round(float(info.get("step_fill_rate", 0)), 4),
                "inventory": env.inventory.tolist(),
            }
            print(json.dumps(step_log), flush=True)

    score = _safe_score(grader.grade(info))

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

    print(f"[END] task={task_id} score={float(score):.4f} steps={step}", flush=True)
    if verbose:
        print(json.dumps(end_log), flush=True)

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
        print(f"  No trained model found for {task_id}, falling back to heuristic")
        return run_heuristic(task_id, seed, verbose)

    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    env = FlattenedWarehouseEnv(task_id=task_id, seed=seed)
    obs, info = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    step = 0

    print(f"[START] task={task_id}", flush=True)
    if verbose:
        print(json.dumps({"event": "START", "task_id": task_id, "agent": "ppo", "episode": 1}), flush=True)

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        print(f"[STEP] step={step} reward={float(reward):.4f}", flush=True)
        if verbose:
            step_log = {
                "event": "STEP",
                "step": step,
                "action_ids": action.tolist() if hasattr(action, 'tolist') else list(action),
                "reward": round(float(reward), 4),
                "done": done,
                "fill_rate": round(float(info.get("step_fill_rate", 0)), 4),
            }
            print(json.dumps(step_log), flush=True)

    score = _safe_score(grader.grade(info))

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

    print(f"[END] task={task_id} score={float(score):.4f} steps={step}", flush=True)
    if verbose:
        print(json.dumps(end_log), flush=True)

    return end_log


def _build_llm_system_prompt(env: WarehouseEnv) -> str:
    """Build a structured, constrained system prompt for the LLM agent.

    Includes environment rules, full action encoding table, product economics,
    and strict JSON output requirements.
    """
    product_names = env.get_product_names()
    num_products = env.num_products

    # Build action table
    action_lines = []
    for idx, qty in enumerate(ORDER_LEVELS):
        label = f"{qty} units" if qty > 0 else "no order"
        action_lines.append(f"  {idx} → {label}")

    emergency_section = ""
    if env.emergency_enabled:
        emergency_section = "\nEmergency orders (3× cost, instant delivery):\n"
        for idx, qty in enumerate(EMERGENCY_ORDER_LEVELS):
            action_idx = idx + len(ORDER_LEVELS)
            label = f"{qty} units (emergency)" if qty > 0 else "no emergency order"
            emergency_section += f"  {action_idx} → {label}\n"

    # Build product economics table
    product_details = []
    for i, pc in enumerate(env.product_configs):
        details = (
            f"  Product {i} ({pc['name']}): "
            f"margin=${pc['margin']}/unit, "
            f"holding=${pc['holding_cost']}/unit/day, "
            f"order_cost=${pc['ordering_cost']}/order"
        )
        if pc['perishable']:
            details += f", perishable (shelf_life={pc['shelf_life']}d, expiry_penalty=${pc['expiry_penalty']})"
        details += f", stockout_penalty=${pc['stockout_penalty']}"
        product_details.append(details)

    capacity_info = f"Capacity: {env.capacity} units shared" if env.capacity else "Capacity: unlimited"
    lt_info = f"Lead time: {env.lead_time_min}–{env.lead_time_max} days"
    reliability_info = f"Supplier reliability: {env.supplier_base_reliability * 100:.0f}%"

    return f"""You are an expert warehouse inventory management agent.

ENVIRONMENT:
- You manage {num_products} product(s): {', '.join(product_names)}
- Each day you choose a reorder action for each product.
- {capacity_info}
- {lt_info}
- {reliability_info}
- Goal: maximize fill rate (meet demand) while minimizing costs.

PRODUCT ECONOMICS:
{chr(10).join(product_details)}

ACTION FORMAT:
Each action is a discrete INDEX (not a quantity). The mapping is:
{chr(10).join(action_lines)}
{emergency_section}
RULES:
1. You MUST respond with ONLY a JSON object: {{"action_ids": [int, ...]}}
2. The array must have exactly {num_products} integer(s).
3. Each integer must be a valid action index (0–{env.actions_per_product - 1}).
4. Do NOT include any explanation, markdown, or extra text.
5. Consider: current inventory, incoming shipments, demand trends, expiry dates, costs.

STRATEGY GUIDELINES:
- If inventory < 2× avg daily demand, consider ordering (index 2–4).
- If inventory > 5× avg daily demand, stop ordering (index 0).
- For perishable goods, prefer smaller, more frequent orders to prevent waste.
- On Thursday/Friday (day 3-4), order extra for weekend demand spikes.
- Factor in lead time: orders placed today arrive in {env.lead_time_min}–{env.lead_time_max} days.
- Emergency orders (if available) should only be used when stockout is imminent.
"""


def _build_llm_user_prompt(
    obs: dict,
    env: WarehouseEnv,
    last_action: Optional[List[int]] = None,
    last_reward: Optional[float] = None,
) -> str:
    """Build a structured observation prompt for the LLM.

    Includes current state, product-level analysis, and optional
    context about the previous step's action and reward.
    """
    product_names = env.get_product_names()
    lines = ["Current warehouse state:"]

    for i, name in enumerate(product_names):
        inv = float(obs["inventory"][i])
        transit = obs["in_transit"][i]
        total_transit = float(np.sum(transit))
        expiry = int(obs["days_to_expiry"][i])
        recent_demand = obs["demand_history"][i]
        recent_nonzero = recent_demand[recent_demand > 0] if isinstance(recent_demand, np.ndarray) else [d for d in recent_demand if d > 0]
        avg_demand = float(np.mean(recent_nonzero)) if len(recent_nonzero) > 0 else 0.0

        lines.append(f"\n  Product {i} ({name}):")
        lines.append(f"    Inventory: {inv:.0f} units")
        lines.append(f"    In transit: {total_transit:.0f} units")
        if avg_demand > 0:
            days_of_stock = (inv + total_transit) / avg_demand
            lines.append(f"    Days of stock: {days_of_stock:.1f}")
        if expiry >= 0:
            lines.append(f"    Days to expiry: {expiry}")
        lines.append(f"    Avg recent demand: {avg_demand:.1f} units/day")

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow = int(obs["day_of_week"])
    lines.append(f"\n  Day: {day_names[dow]} (index {dow})")

    storage = obs["storage_used"]
    if hasattr(storage, '__len__'):
        storage = float(storage[0])
    lines.append(f"  Storage used: {storage*100:.1f}%")

    # Add context from previous step
    if last_action is not None and last_reward is not None:
        lines.append(f"\n  Previous action: {last_action}")
        lines.append(f"  Previous reward: {last_reward:.4f}")

    lines.append(f'\nRespond with: {{"action_ids": [one index per product]}}')
    return "\n".join(lines)


def _parse_llm_response(reply: str, num_products: int, max_action: int) -> Optional[list]:
    """Parse and validate LLM response. Returns action_ids list or None on failure.

    Handles common LLM output quirks:
    - Markdown code fences
    - Extra text around JSON
    - Plain arrays without the action_ids key
    - Alternative key names (actions, action)
    """
    reply = reply.strip()

    # Remove markdown code fences if present
    if reply.startswith("```"):
        lines = reply.split("\n")
        reply = "\n".join(l for l in lines if not l.startswith("```"))
        reply = reply.strip()

    try:
        parsed = json.loads(reply)
    except json.JSONDecodeError:
        # Try to find a JSON object in the text
        import re
        match = re.search(r'\{[^}]+\}', reply)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                # Try broader match for nested objects
                pass
        else:
            parsed = None

        if parsed is None:
            # Try parsing as a plain array
            match = re.search(r'\[[\d,\s]+\]', reply)
            if match:
                try:
                    arr = json.loads(match.group())
                    parsed = {"action_ids": arr}
                except json.JSONDecodeError:
                    return None
            else:
                return None

    # Extract action_ids from various key names
    if isinstance(parsed, dict):
        action_ids = (
            parsed.get("action_ids")
            or parsed.get("actions")
            or parsed.get("action")
            or None
        )
    elif isinstance(parsed, list):
        action_ids = parsed
    else:
        return None

    if not isinstance(action_ids, list):
        return None
    if len(action_ids) != num_products:
        return None

    # Validate and clamp each index
    validated = []
    for aid in action_ids:
        try:
            aid = int(aid)
            aid = max(0, min(aid, max_action - 1))
            validated.append(aid)
        except (ValueError, TypeError):
            return None

    return validated


def run_llm(task_id: str, seed: int = 42, verbose: bool = True, max_retries: int = 2) -> dict:
    """Run an LLM-based agent using the OpenAI client on a task.

    Features:
    - Structured system prompt with environment rules, economics, and legal actions
    - Strict JSON output format: {"action_ids": [int, ...]}
    - Robust parsing with up to max_retries on parse failure before heuristic fallback
    - Rolling context: previous action and reward passed to each step
    - Action index validation and clamping
    """
    config = load_task_config(task_id)
    grader = get_grader(config)

    client = OpenAI(
        base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ.get("API_KEY", os.environ.get("OPENAI_API_KEY", "dummy-key"))
    )

    env = WarehouseEnv(task_id=task_id, seed=seed)
    # Create a fallback heuristic agent for when LLM fails
    fallback_agent = HeuristicAgent(env)

    system_prompt = _build_llm_system_prompt(env)

    obs, info = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    step = 0
    llm_failures = 0
    last_action = None
    last_reward = None

    print(f"[START] task={task_id}", flush=True)
    if verbose:
        print(json.dumps({"event": "START", "task_id": task_id, "agent": "llm", "episode": 1}), flush=True)

    while not done:
        user_prompt = _build_llm_user_prompt(obs, env, last_action, last_reward)
        action_ids = None

        # Try up to max_retries + 1 times
        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=100,
                    temperature=0.1,  # Low temperature for consistent structured output
                )
                reply = response.choices[0].message.content.strip()
                action_ids = _parse_llm_response(reply, env.num_products, env.actions_per_product)

                if action_ids is not None:
                    break  # Success

                if verbose:
                    print(
                        f"  [LLM] Parse failed (attempt {attempt + 1}/{max_retries + 1}): {reply[:80]}",
                        flush=True,
                    )
            except Exception as e:
                if verbose:
                    print(f"  [LLM] API error (attempt {attempt + 1}): {e}", flush=True)

        if action_ids is None:
            llm_failures += 1
            if verbose:
                print(f"  [LLM] All attempts failed, using heuristic fallback", flush=True)

        # Fallback to heuristic if LLM fails
        if action_ids is not None:
            action = np.array(action_ids, dtype=np.int64)
        else:
            action = fallback_agent.act(obs)

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # Track for rolling context
        last_action = action.tolist() if hasattr(action, 'tolist') else list(action)
        last_reward = float(reward)

        print(f"[STEP] step={step} reward={float(reward):.4f}", flush=True)
        if verbose:
            step_log = {
                "event": "STEP",
                "step": step,
                "action_ids": last_action,
                "reward": round(float(reward), 4),
                "done": done,
                "fill_rate": round(float(info.get("step_fill_rate", 0)), 4),
                "source": "llm" if action_ids is not None else "heuristic_fallback",
            }
            print(json.dumps(step_log), flush=True)

    score = _safe_score(grader.grade(info))

    end_log = {
        "event": "END",
        "task_id": task_id,
        "agent": "llm",
        "total_reward": round(total_reward, 4),
        "score": round(score, 4),
        "fill_rate": round(float(info.get("fill_rate", 0)), 4),
        "waste_rate": round(float(info.get("waste_rate", 0)), 4),
        "total_revenue": round(float(info.get("total_revenue", 0)), 2),
        "total_holding_cost": round(float(info.get("total_holding_cost", 0)), 2),
        "total_ordering_cost": round(float(info.get("total_ordering_cost", 0)), 2),
        "llm_failures": llm_failures,
    }

    print(f"[END] task={task_id} score={float(score):.4f} steps={step} llm_failures={llm_failures}", flush=True)
    if verbose:
        print(json.dumps(end_log), flush=True)

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
        help="Path to trained PPO model directory. If omitted, runs heuristic baseline.",
    )
    parser.add_argument(
        "--use-llm", action="store_true", help="Run LLM agent using configured API endpoints."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--num-seeds", type=int, default=1,
        help="Number of seeds for multi-seed evaluation (default: 1)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Only print final results, no step logs"
    )
    args = parser.parse_args()

    # Auto-detect OpenEnv LiteLLM proxy and force LLM mode
    if os.environ.get("API_KEY") or os.environ.get("API_BASE_URL"):
        args.use_llm = True

    tasks = (
        ["task1_single_product", "task2_multi_product", "task3_nonstationary"]
        if args.task == "all"
        else [args.task]
    )

    agent_name = "LLM (OpenAI API)" if args.use_llm else ("PPO" if args.model_dir else "Heuristic Baseline")

    print(f"\n{'='*60}")
    print("Warehouse Inventory Management — Inference")
    print(f"Agent: {agent_name}")
    print(f"Seeds: {args.num_seeds} (starting from {args.seed})")
    print(f"{'='*60}\n")

    all_results = {}
    for task_id in tasks:
        task_scores = []
        task_results = []
        for s in range(args.num_seeds):
            current_seed = args.seed + s
            print(f"\n--- {task_id} (seed={current_seed}) ---")
            if args.use_llm:
                result = run_llm(task_id, current_seed, not args.quiet)
            elif args.model_dir:
                result = run_ppo(task_id, args.model_dir, current_seed, not args.quiet)
            else:
                result = run_heuristic(task_id, current_seed, not args.quiet)
            task_scores.append(result["score"])
            task_results.append(result)

        all_results[task_id] = {
            "results": task_results,
            "avg_score": float(np.mean(task_scores)),
            "std_score": float(np.std(task_scores)),
        }

    # Summary table
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Task':<25} {'Avg Score':>10} {'Std':>8} {'Fill Rate':>10} {'Reward':>10}")
    print("-" * 65)
    for tid, r in all_results.items():
        last = r["results"][-1]
        print(
            f"{tid:<25} {r['avg_score']:>10.4f} {r['std_score']:>8.4f} "
            f"{last['fill_rate']:>10.4f} {last['total_reward']:>10.4f}"
        )
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()