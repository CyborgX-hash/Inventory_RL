"""
Graders for the three task difficulty levels.

Each grader takes an episode info dict from the environment and produces
a final score in the open interval (0, 1), strictly excluding the boundaries.

Epsilon clamping ensures scores are never exactly 0.0 or 1.0,
which some hackathon validators reject.
"""


import numpy as np

# Safe score boundaries — validators may reject exact 0 or 1
SCORE_EPS = 1e-3
SCORE_MIN = SCORE_EPS
SCORE_MAX = 1.0 - SCORE_EPS


def _safe_score(score: float) -> float:
    """Clamp a raw score to the safe open interval (SCORE_MIN, SCORE_MAX)."""
    return float(np.clip(score, SCORE_MIN, SCORE_MAX))


class Task1Grader:
    """Easy: Single product, stable demand.

    Score = blend of fill_rate and holding_cost efficiency.
    Full marks if fill_rate >= 0.90 AND avg holding cost <= threshold.
    Partial credit for being close.
    """

    def __init__(self, config: dict):
        self.fill_rate_target = config.get("fill_rate_target", 0.90)
        self.holding_cost_threshold = config.get("holding_cost_threshold", 30.0)

    def grade(self, info: dict) -> float:
        fill_rate = info.get("fill_rate", 0.0)
        total_holding = info.get("total_holding_cost", 0.0)
        steps = max(info.get("step_count", 1), 1)
        avg_holding = total_holding / steps

        # Fill rate score: linear from 0 at fill_rate=0 to 1 at target
        fr_score = np.clip(fill_rate / max(self.fill_rate_target, 0.01), 0.0, 1.0)

        # Holding cost score: 1.0 if below threshold, linear decay to 0 at 3× threshold
        if avg_holding <= self.holding_cost_threshold:
            hc_score = 1.0
        else:
            hc_score = np.clip(
                1.0
                - (avg_holding - self.holding_cost_threshold)
                / (self.holding_cost_threshold * 2.0),
                0.0,
                1.0,
            )

        # Combined score: 70% fill rate (primary objective), 30% holding cost
        score = 0.7 * fr_score + 0.3 * hc_score
        return _safe_score(score)


class Task2Grader:
    """Medium: Multi-product with constraints.

    Weighted score of fill_rate, waste_rate, inventory_turnover.
    """

    def __init__(self, config: dict):
        weights = config.get("weights", {})
        self.w_fill = weights.get("fill_rate", 0.5)
        self.w_waste = weights.get("waste_rate", 0.3)
        self.w_turnover = weights.get("inventory_turnover", 0.2)

    def grade(self, info: dict) -> float:
        fill_rate = info.get("fill_rate", 0.0)
        waste_rate = info.get("waste_rate", 0.0)
        turnover = info.get("inventory_turnover", 0.0)

        # Fill rate score: linear 0→1 (full range, no cliff)
        fr_score = np.clip(fill_rate, 0.0, 1.0)

        # Waste score: 1.0 if no waste, 0 if waste_rate >= 0.5 (more lenient)
        waste_score = np.clip(1.0 - waste_rate / 0.5, 0.0, 1.0)

        # Turnover score: higher is better, normalized by a reasonable max
        turnover_score = np.clip(turnover / 8.0, 0.0, 1.0)

        score = (
            self.w_fill * fr_score
            + self.w_waste * waste_score
            + self.w_turnover * turnover_score
        )
        return _safe_score(score)


class Task3Grader:
    """Hard: Non-stationary demand + supplier uncertainty.

    Score based on:
    - Fill rate (primary signal, 40%)
    - Profitability relative to a dynamic baseline (40%)
    - Penalty for overusing emergency reorders (20% budget)

    The baseline profit is auto-computed from the heuristic agent
    if not already set.
    """

    def __init__(self, config: dict):
        self.baseline_profit = config.get("baseline_profit_margin", 0.0)
        self.emergency_penalty = config.get("emergency_overuse_penalty", 0.05)
        self.emergency_threshold = config.get("emergency_threshold", 5)
        self._baseline_computed = self.baseline_profit > 0

    def set_baseline_profit(self, profit: float):
        """Set from heuristic agent evaluation."""
        self.baseline_profit = profit
        self._baseline_computed = True

    def _ensure_baseline(self):
        """Auto-compute baseline profit from heuristic if not already set."""
        if self._baseline_computed:
            return
        try:
            from baseline.heuristic_agent import HeuristicAgent
            from environment.warehouse_env import WarehouseEnv

            env = WarehouseEnv("task3_nonstationary", seed=0)
            agent = HeuristicAgent(env)
            results = agent.evaluate(num_episodes=5, seed=0)
            self.baseline_profit = max(results["avg_profit"], 1.0)
            self._baseline_computed = True
        except Exception:
            # Fallback: use a reasonable estimate
            self.baseline_profit = 80000.0
            self._baseline_computed = True

    def grade(self, info: dict) -> float:
        self._ensure_baseline()

        fill_rate = info.get("fill_rate", 0.0)
        revenue = info.get("total_revenue", 0.0)
        holding = info.get("total_holding_cost", 0.0)
        ordering = info.get("total_ordering_cost", 0.0)
        expired_cost = info.get("total_units_expired", 0.0) * 8.0
        profit = revenue - holding - ordering - expired_cost

        # --- Fill rate score (40% weight) ---
        fr_score = np.clip(fill_rate, 0.0, 1.0)

        # --- Profit score (40% weight) ---
        # 0.5 if matching baseline, 1.0 if 1.5× baseline
        if self.baseline_profit > 0:
            ratio = profit / self.baseline_profit
            profit_score = np.clip(ratio, 0.0, 1.0)
        elif profit > 0:
            profit_score = 0.6
        else:
            profit_score = np.clip(0.2 + profit / 100000.0, 0.0, 0.2)

        # --- Emergency overuse penalty (from the 20% budget) ---
        emergency_orders = info.get("total_emergency_orders", 0)
        emergency_score = 1.0
        if emergency_orders > self.emergency_threshold:
            excess = emergency_orders - self.emergency_threshold
            emergency_score = np.clip(1.0 - excess * self.emergency_penalty, 0.0, 1.0)

        score = 0.4 * fr_score + 0.4 * profit_score + 0.2 * emergency_score
        return _safe_score(score)


def get_grader(config: dict):
    """Factory: return the appropriate grader based on config type."""
    grader_config = config.get("grader", {})
    grader_type = grader_config.get("type", "task1")

    if grader_type == "task1":
        return Task1Grader(grader_config)
    elif grader_type == "task2":
        return Task2Grader(grader_config)
    elif grader_type == "task3":
        return Task3Grader(grader_config)
    else:
        raise ValueError(f"Unknown grader type: {grader_type}")
