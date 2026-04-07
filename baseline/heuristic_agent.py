"""
Heuristic baseline agent using a Reorder-Point (s, S) policy.

For each product:
  - Compute reorder point s = avg_demand × lead_time + safety_stock
  - Compute order-up-to level S = s + avg_demand × lead_time
  - When inventory[i] <= s, order up to S (rounded to nearest valid quantity)

Enhanced with:
  - Supplier reliability compensation (order more when reliability < 1.0)
  - Emergency order support (when inventory critically low)
  - Weekend demand anticipation

This serves as the performance baseline that the RL agent (Task 3) must beat.
"""

import numpy as np
from typing import List, Optional

from environment.warehouse_env import WarehouseEnv, ORDER_LEVELS, EMERGENCY_ORDER_LEVELS


class HeuristicAgent:
    """(s, S) reorder-point policy with safety stock and reliability compensation."""

    def __init__(self, env: WarehouseEnv, safety_factor: float = 2.0):
        """
        Args:
            env: WarehouseEnv instance
            safety_factor: Multiplier for safety stock (higher = more conservative)
        """
        self.env = env
        self.num_products = env.num_products
        self.safety_factor = safety_factor
        self.order_levels = ORDER_LEVELS
        self.emergency_enabled = env.emergency_enabled

        # Precompute initial estimates from config
        self.avg_demands = np.array(env.demand_sim.means, dtype=np.float64)
        self.demand_stds = np.array(env.demand_sim.stds, dtype=np.float64)
        self.avg_lead_time = (env.lead_time_min + env.lead_time_max) / 2.0
        self.max_lead_time = env.lead_time_max
        self.supplier_reliability = env.supplier_base_reliability

        # Compensate for supplier unreliability:
        # If only 70% of orders are fully filled, order ~1/0.7 = 1.43× more
        self.reliability_multiplier = 1.0 / max(self.supplier_reliability, 0.5)

        # Reorder point: uses MAX lead time (worst case) + safety stock
        safety_stock = self.safety_factor * self.demand_stds * np.sqrt(self.max_lead_time)
        self.reorder_point = (
            self.avg_demands * self.avg_lead_time * self.reliability_multiplier
            + safety_stock
        )

        # Order-up-to level: cover demand for max lead time + buffer
        self.order_up_to = (
            self.reorder_point
            + self.avg_demands * self.avg_lead_time * self.reliability_multiplier
        )

        # Critical level: below this, use emergency orders
        self.critical_level = self.avg_demands * 1.5  # ~1.5 days of demand

    def _snap_to_order_level(self, qty: float, emergency: bool = False) -> int:
        """Round desired quantity to nearest valid order level (rounding UP).

        Returns action index. For emergency orders, offset by len(ORDER_LEVELS).
        """
        levels = EMERGENCY_ORDER_LEVELS if emergency else self.order_levels
        if qty <= 0:
            return len(ORDER_LEVELS) if emergency else 0

        # Round UP to ensure we don't under-order
        best = 0
        best_level = 0
        for idx, level in enumerate(levels):
            if level >= qty:
                best = idx
                best_level = level
                break
            best = idx
            best_level = level

        if emergency:
            return best + len(ORDER_LEVELS)
        return best

    def act(self, obs: dict) -> np.ndarray:
        """Select order quantities based on (s, S) policy with enhancements.

        Returns:
            Action array of indices (into ORDER_LEVELS or extended emergency space)
        """
        inventory = obs["inventory"]
        in_transit = obs["in_transit"]
        demand_hist = obs["demand_history"]
        day_of_week = obs["day_of_week"]
        if isinstance(day_of_week, np.ndarray):
            day_of_week = int(day_of_week)

        actions = np.zeros(self.num_products, dtype=np.int64)

        for i in range(self.num_products):
            # Effective inventory = on-hand + in-transit
            effective = inventory[i] + np.sum(in_transit[i])

            # Update demand estimate from recent history
            recent = demand_hist[i]
            recent_nonzero = recent[recent > 0]
            if len(recent_nonzero) >= 3:
                avg_demand = np.mean(recent_nonzero)
                demand_std = max(np.std(recent_nonzero), 1.0)
            else:
                avg_demand = self.avg_demands[i]
                demand_std = self.demand_stds[i]

            # Weekend anticipation: if Thursday/Friday, bump demand estimate
            if day_of_week in (3, 4):  # Thursday, Friday
                avg_demand *= 1.3

            # Dynamic reorder point with reliability compensation
            safety = self.safety_factor * demand_std * np.sqrt(self.max_lead_time)
            s = avg_demand * self.avg_lead_time * self.reliability_multiplier + safety
            S = s + avg_demand * self.avg_lead_time * self.reliability_multiplier

            # Emergency order: if inventory critically low and emergency enabled
            if (
                self.emergency_enabled
                and inventory[i] < self.critical_level[i]
                and np.sum(in_transit[i]) < avg_demand  # nothing significant coming
            ):
                desired_qty = S - effective
                actions[i] = self._snap_to_order_level(desired_qty, emergency=True)
            elif effective <= s:
                desired_qty = (S - effective) * self.reliability_multiplier
                actions[i] = self._snap_to_order_level(desired_qty, emergency=False)
            else:
                actions[i] = 0  # No order needed

        return actions

    def evaluate(self, num_episodes: int = 10, seed: int = 42) -> dict:
        """Run evaluation episodes and return average metrics.

        Returns:
            dict with avg_score, avg_fill_rate, avg_profit, etc.
        """
        from environment.graders import get_grader

        grader = get_grader(self.env.config)
        results = {
            "scores": [],
            "fill_rates": [],
            "profits": [],
            "waste_rates": [],
        }

        for ep in range(num_episodes):
            obs, info = self.env.reset(seed=seed + ep)
            done = False
            total_reward = 0.0

            while not done:
                action = self.act(obs)
                obs, reward, done, truncated, info = self.env.step(action)
                total_reward += reward

            score = grader.grade(info)
            profit = info["total_revenue"] - info["total_holding_cost"] - info["total_ordering_cost"]

            results["scores"].append(score)
            results["fill_rates"].append(info["fill_rate"])
            results["profits"].append(profit)
            results["waste_rates"].append(info["waste_rate"])

        return {
            "avg_score": float(np.mean(results["scores"])),
            "avg_fill_rate": float(np.mean(results["fill_rates"])),
            "avg_profit": float(np.mean(results["profits"])),
            "avg_waste_rate": float(np.mean(results["waste_rates"])),
            "scores": results["scores"],
        }
