"""
Fixed-quantity baseline agent — always orders the same amount.

This is the simplest possible policy: order a fixed quantity every step.
Serves as a lower bound to demonstrate the value of adaptive policies
like the (s,S) heuristic and PPO agent.
"""


import numpy as np

from environment.warehouse_env import ORDER_LEVELS, WarehouseEnv


class FixedQuantityAgent:
    """Always orders the same fixed quantity for every product every step.

    Args:
        env: WarehouseEnv instance.
        action_index: Which ORDER_LEVELS index to use (default: 2 → 10 units).
    """

    def __init__(self, env: WarehouseEnv, action_index: int = 2):
        self.env = env
        self.num_products = env.num_products
        self.action_index = min(action_index, len(ORDER_LEVELS) - 1)

    def act(self, obs: dict) -> np.ndarray:
        """Return the same fixed action every step."""
        return np.full(self.num_products, self.action_index, dtype=np.int64)

    def evaluate(self, num_episodes: int = 10, seed: int = 42) -> dict:
        """Run evaluation episodes and return average metrics."""
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

            while not done:
                action = self.act(obs)
                obs, reward, done, truncated, info = self.env.step(action)

            score = grader.grade(info)
            profit = (
                info["total_revenue"]
                - info["total_holding_cost"]
                - info["total_ordering_cost"]
            )

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
