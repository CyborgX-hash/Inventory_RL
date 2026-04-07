"""
Demand simulator supporting stationary, seasonal, and non-stationary demand patterns.
"""

import numpy as np
from typing import Dict, List, Optional


class DemandSimulator:
    """Generates stochastic customer demand for warehouse products.

    Supports three demand models:
      - stationary:     Normal(mean, std), no time dependence
      - seasonal:       Normal with day-of-week multiplier (weekend spike)
      - nonstationary:  Seasonal + monthly trend + random demand shocks +
                        cross-product substitution effects
    """

    def __init__(self, config: dict, num_products: int, rng: np.random.Generator):
        self.config = config
        self.num_products = num_products
        self.rng = rng

        self.demand_type = config.get("type", "stationary")

        # Per-product parameters
        if self.demand_type == "stationary":
            # Single product: config has top-level mean/std
            self.means = [config["mean"]] * num_products
            self.stds = [config["std"]] * num_products
        else:
            # Multi-product: config["products"] list
            products = config["products"]
            if "base_mean" in products[0]:
                self.means = [p["base_mean"] for p in products]
            else:
                self.means = [p["mean"] for p in products]
            self.stds = [p["std"] for p in products]

        # Seasonality
        self.seasonality = config.get("seasonality", False)
        self.weekend_multiplier = config.get("weekend_multiplier", 1.0)

        # Non-stationary parameters
        self.monthly_trend_amplitude = config.get("monthly_trend_amplitude", 0.0)
        self.demand_shock_prob = config.get("demand_shock_probability", 0.0)
        self.substitution_factor = config.get("substitution_factor", 0.0)

    def _seasonal_multiplier(self, day_of_week: int) -> float:
        """Return demand multiplier based on day of week. Sat=5, Sun=6."""
        if self.seasonality and day_of_week >= 5:
            return self.weekend_multiplier
        return 1.0

    def _monthly_trend(self, global_step: int) -> float:
        """Sinusoidal monthly trend: amplitude × sin(2π × step / 30)."""
        if self.monthly_trend_amplitude > 0:
            return self.monthly_trend_amplitude * np.sin(
                2.0 * np.pi * global_step / 30.0
            )
        return 0.0

    def generate(
        self,
        day_of_week: int,
        global_step: int,
        current_inventory: Optional[List[float]] = None,
        product_configs: Optional[List[dict]] = None,
    ) -> np.ndarray:
        """Generate demand for all products for the current day.

        Args:
            day_of_week: 0-6 (Mon-Sun)
            global_step: Absolute step number in the episode
            current_inventory: Current stock levels (for substitution calculation)
            product_configs: Product config dicts (for substitution relationships)

        Returns:
            np.ndarray of shape (num_products,) with non-negative integer demands
        """
        seasonal_mult = self._seasonal_multiplier(day_of_week)
        trend_offset = self._monthly_trend(global_step)

        demands = np.zeros(self.num_products)

        for i in range(self.num_products):
            base = self.means[i]

            # Apply trend (non-stationary)
            adjusted_mean = base * (1.0 + trend_offset)

            # Apply seasonality
            adjusted_mean *= seasonal_mult

            # Sample from normal distribution
            demand = self.rng.normal(adjusted_mean, self.stds[i])

            # Demand shock (rare large spike)
            if self.demand_shock_prob > 0 and self.rng.random() < self.demand_shock_prob:
                demand *= 2.0

            demands[i] = max(0.0, demand)

        # Cross-product substitution (Task 3)
        if (
            self.substitution_factor > 0
            and current_inventory is not None
            and product_configs is not None
        ):
            demands = self._apply_substitution(
                demands, current_inventory, product_configs
            )

        # Round to integers
        return np.maximum(0, np.round(demands)).astype(int)

    def _apply_substitution(
        self,
        demands: np.ndarray,
        current_inventory: List[float],
        product_configs: List[dict],
    ) -> np.ndarray:
        """When product A stocks out, a fraction of its demand shifts to substitute B.

        If product B has `substitute_for: A_index`, then when A has a stockout,
        B picks up `substitution_factor` fraction of A's unmet demand.
        Additionally, selling A reduces demand for B (competitive effect).
        """
        adjusted = demands.copy()

        for i, pc in enumerate(product_configs):
            sub_target = pc.get("substitute_for")
            if sub_target is not None and sub_target is not False:
                target_idx = int(sub_target)
                # Unmet demand from target product
                unmet = max(0.0, demands[target_idx] - current_inventory[target_idx])
                # A fraction of unmet demand shifts to this product
                adjusted[i] += self.substitution_factor * unmet

        return adjusted
