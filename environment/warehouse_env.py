"""
Warehouse inventory management environment — Gymnasium compatible.

Manages inventory, in-transit orders, perishability, supplier unreliability,
and emergency reorder logic across all three task difficulty levels.
"""

import os
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from environment.demand_simulator import DemandSimulator
from environment.graders import SCORE_MAX, SCORE_MIN
from environment.models import (
    ActionChoice,
    ProductActionMetadata,
    WarehouseState,
)

# Valid per-product order quantities
ORDER_LEVELS = [0, 5, 10, 20, 50, 100]
# For Task 3, the agent may also choose an emergency order.
# We encode this by doubling the action space: first 6 = normal, next 6 = emergency
EMERGENCY_ORDER_LEVELS = [0, 5, 10, 20, 50, 100]

TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tasks")


@dataclass(frozen=True)
class RewardConfig:
    """Configurable coefficients for the reward function.

    Extracted from magic numbers so reward shaping can be tuned
    and documented in one place.

    Reward = revenue
             − holding_cost × holding_weight
             − order_cost × ordering_weight
             − expiry_penalty
             − stockout_penalty × stockout_multiplier
             + fill_rate_shaping_reward
    """

    holding_weight: float = 0.3
    """Downscale holding cost vs revenue so it doesn't dominate."""

    ordering_weight: float = 0.5
    """Frequency-of-ordering penalty: encourages batching."""

    stockout_multiplier: float = 3.0
    """Extra multiplier on per-unit stockout penalty from config (strong signal)."""

    fill_rate_shape_weight: float = 0.5
    """Weight of quadratic fill-rate shaping term relative to expected revenue."""


def load_task_config(task_id: str) -> dict:
    """Load a task YAML config by its id."""
    mapping = {
        "task1_single_product": "task1_easy.yaml",
        "task2_multi_product": "task2_medium.yaml",
        "task3_nonstationary": "task3_hard.yaml",
    }
    filename = mapping.get(task_id)
    if filename is None:
        raise ValueError(f"Unknown task_id: {task_id}")
    path = os.path.join(TASKS_DIR, filename)
    with open(path) as f:
        return yaml.safe_load(f)


class WarehouseEnv(gym.Env):
    """Multi-product warehouse inventory management environment.

    Observation (dict):
        inventory        : (N,)      current stock per product
        in_transit       : (N, max_lt) units arriving in d days per product
        days_to_expiry   : (N,)      shelf life remaining (-1 = non-perishable)
        demand_history   : (N, 7)    last 7 days of demand
        storage_used     : ()        fraction of capacity in use
        day_of_week      : ()        0-6
        supplier_reliability : (N,)  recent on-time delivery rate

    Action (MultiDiscrete):
        Per product: index into ORDER_LEVELS (or extended with emergency levels)

    Action Encoding:
        Index 0 → order 0 units
        Index 1 → order 5 units
        Index 2 → order 10 units
        Index 3 → order 20 units
        Index 4 → order 50 units
        Index 5 → order 100 units
        (Task 3 only) Index 6–11 → emergency orders of same quantities
    """

    metadata = {"render_modes": []}

    def __init__(self, task_id: str = "task1_single_product", seed: int = 42):
        super().__init__()
        self.task_id = task_id
        self.config = load_task_config(task_id)
        self.rng = np.random.default_rng(seed)

        # --- Products ---
        self.product_configs: list[dict] = self.config["products"]
        self.num_products = len(self.product_configs)

        # --- Supply ---
        supply = self.config["supply"]
        self.lead_time_min = supply["lead_time_min"]
        self.lead_time_max = supply["lead_time_max"]
        self.max_lead_time = self.lead_time_max
        self.supplier_base_reliability = supply["reliability"]
        self.partial_fill_min = supply.get("partial_fill_min", 1.0)

        # --- Emergency reorder ---
        emerg = self.config.get("emergency_reorder", {})
        self.emergency_enabled = emerg.get("enabled", False)
        self.emergency_cost_mult = emerg.get("cost_multiplier", 3.0)
        self.emergency_lead_time = emerg.get("lead_time", 0)

        # --- Warehouse ---
        wh = self.config["warehouse"]
        self.capacity = wh.get("capacity")  # None = unlimited

        # --- Reward ---
        self.reward_config = RewardConfig()

        # --- Task params ---
        self.max_steps = self.config["max_steps"]

        # --- Demand simulator ---
        self.demand_sim = DemandSimulator(
            self.config["demand"], self.num_products, self.rng
        )

        # --- Action space ---
        if self.emergency_enabled:
            self.actions_per_product = len(ORDER_LEVELS) + len(EMERGENCY_ORDER_LEVELS)
        else:
            self.actions_per_product = len(ORDER_LEVELS)

        self.action_space = spaces.MultiDiscrete(
            [self.actions_per_product] * self.num_products
        )

        # --- Observation space ---
        big = 1e6
        self.observation_space = spaces.Dict(
            {
                "inventory": spaces.Box(
                    0, big, shape=(self.num_products,), dtype=np.float32
                ),
                "in_transit": spaces.Box(
                    0,
                    big,
                    shape=(self.num_products, self.max_lead_time),
                    dtype=np.float32,
                ),
                "days_to_expiry": spaces.Box(
                    -1, 365, shape=(self.num_products,), dtype=np.int32
                ),
                "demand_history": spaces.Box(
                    0, big, shape=(self.num_products, 7), dtype=np.float32
                ),
                "storage_used": spaces.Box(0, 1.0, shape=(1,), dtype=np.float32),
                "day_of_week": spaces.Discrete(7),
                "supplier_reliability": spaces.Box(
                    0, 1, shape=(self.num_products,), dtype=np.float32
                ),
            }
        )

        # Internal state (set in reset)
        self._reset_state()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_state(self):
        """Initialize all internal tracking variables."""
        self.inventory = np.array(
            [p["initial_inventory"] for p in self.product_configs], dtype=np.float64
        )
        self.in_transit = np.zeros(
            (self.num_products, self.max_lead_time), dtype=np.float64
        )
        self.days_to_expiry = np.array(
            [p.get("shelf_life") or -1 for p in self.product_configs], dtype=np.int32
        )
        self.expiry_tracker: list[list[tuple[float, int]]] = [
            [] for _ in range(self.num_products)
        ]
        # Initialize expiry batches for initial inventory
        for i, pc in enumerate(self.product_configs):
            if pc["perishable"]:
                self.expiry_tracker[i].append(
                    (float(self.inventory[i]), pc["shelf_life"])
                )

        self.demand_history = np.zeros(
            (self.num_products, 7), dtype=np.float64
        )
        self.step_count = 0
        self.day_of_week = 0

        # Supplier reliability tracking (rolling window)
        self.supplier_delivery_history: list[list[float]] = [
            [1.0] * 5 for _ in range(self.num_products)
        ]

        # Episode accumulators for grading
        self.total_units_demanded = np.zeros(self.num_products)
        self.total_units_sold = np.zeros(self.num_products)
        self.total_units_expired = np.zeros(self.num_products)
        self.total_holding_cost = 0.0
        self.total_ordering_cost = 0.0
        self.total_revenue = 0.0
        self.total_stockout_units = np.zeros(self.num_products)
        self.total_emergency_orders = 0
        self.total_cost_of_goods = 0.0

        # Rolling inventory sum for accurate turnover calculation
        self._inventory_sum_over_steps = 0.0
        # Service level tracking: number of days with fill_rate >= 0.95
        self._high_service_days = 0

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, Any], dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.demand_sim.rng = self.rng
        self._reset_state()
        return self._get_obs(), self._get_info()

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------
    def decode_action(self, action_index: int, product_index: int) -> tuple[int, bool]:
        """Decode a single action index into (order_quantity, is_emergency).

        Args:
            action_index: Index into the action space for this product.
            product_index: Which product this action is for.

        Returns:
            (quantity, is_emergency) tuple.
        """
        if self.emergency_enabled and action_index >= len(ORDER_LEVELS):
            qty = EMERGENCY_ORDER_LEVELS[action_index - len(ORDER_LEVELS)]
            return qty, True
        else:
            qty = ORDER_LEVELS[min(action_index, len(ORDER_LEVELS) - 1)]
            return qty, False

    def validate_action(self, action: np.ndarray) -> np.ndarray:
        """Validate and clamp action indices to legal range.

        Args:
            action: Array of action indices, one per product.

        Returns:
            Clamped action array with all indices in [0, actions_per_product).
        """
        action = np.asarray(action, dtype=np.int64)
        if len(action) != self.num_products:
            raise ValueError(
                f"Expected {self.num_products} action indices, got {len(action)}"
            )
        return np.clip(action, 0, self.actions_per_product - 1)

    def get_product_names(self) -> list[str]:
        """Return human-readable product names from config."""
        return [p["name"] for p in self.product_configs]

    def get_legal_actions(self) -> list[ProductActionMetadata]:
        """Return structured metadata about legal actions for each product.

        This is the single source of truth for the frontend and LLM agents.
        """
        result = []
        for i, pc in enumerate(self.product_configs):
            choices = []
            for idx, qty in enumerate(ORDER_LEVELS):
                label = f"{qty} units" if qty > 0 else "No order"
                choices.append(
                    ActionChoice(
                        index=idx,
                        order_quantity=qty,
                        is_emergency=False,
                        label=label,
                    )
                )
            if self.emergency_enabled:
                for idx, qty in enumerate(EMERGENCY_ORDER_LEVELS):
                    action_idx = idx + len(ORDER_LEVELS)
                    label = (
                        f"{qty} units (emergency)" if qty > 0 else "No emergency order"
                    )
                    choices.append(
                        ActionChoice(
                            index=action_idx,
                            order_quantity=qty,
                            is_emergency=True,
                            label=label,
                        )
                    )
            result.append(
                ProductActionMetadata(
                    product_index=i,
                    product_name=pc["name"],
                    legal_actions=choices,
                )
            )
        return result

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict]:
        """Execute one day of warehouse operations.

        Args:
            action: Array of action indices (one per product). Each index
                    maps to a quantity via ORDER_LEVELS. NOT raw quantities.

        Order of operations:
        1. Receive in-transit shipments that arrive today
        2. Decode action → place new orders (normal and/or emergency)
        3. Generate customer demand, fulfill from inventory
        4. Expire perishable goods
        5. Compute reward
        6. Advance day
        """
        action = self.validate_action(action)

        # ---- 1. Receive arriving shipments ----
        self._receive_shipments()

        # ---- 2. Decode and place orders ----
        order_costs, emergency_count = self._place_orders(action)

        self.total_ordering_cost += order_costs
        self.total_emergency_orders += emergency_count

        # ---- 3. Generate demand and fulfill ----
        demand, units_sold, stockout_units, revenue = self._fulfill_demand()

        self.total_units_demanded += demand
        self.total_units_sold += units_sold
        self.total_stockout_units += stockout_units
        self.total_revenue += revenue

        # Update demand history (rolling window of 7 days)
        self.demand_history = np.roll(self.demand_history, -1, axis=1)
        self.demand_history[:, -1] = demand

        # ---- 4. Expire perishable goods ----
        units_expired = self._expire_perishables()
        self.total_units_expired += units_expired

        # ---- 5. Compute reward ----
        holding_cost = self._compute_holding_cost()
        self.total_holding_cost += holding_cost

        reward, raw_reward = self._compute_reward(
            revenue,
            holding_cost,
            order_costs,
            units_expired,
            stockout_units,
            demand,
            units_sold,
        )

        # ---- 6. Advance day ----
        self.step_count += 1
        self.day_of_week = (self.day_of_week + 1) % 7

        # Track rolling inventory for turnover & service level
        self._inventory_sum_over_steps += float(np.sum(self.inventory))
        step_fill_rate = float(units_sold.sum() / max(demand.sum(), 1.0))
        if step_fill_rate >= 0.95:
            self._high_service_days += 1

        done = self.step_count >= self.max_steps

        rc = self.reward_config
        info = self._get_info()
        info.update(
            {
                "step_revenue": revenue,
                "step_holding_cost": holding_cost,
                "step_order_cost": order_costs,
                "step_expiry_penalty": float(
                    sum(
                        units_expired[i] * self.product_configs[i]["expiry_penalty"]
                        for i in range(self.num_products)
                    )
                ),
                "step_stockout_penalty": float(
                    sum(
                        stockout_units[i]
                        * self.product_configs[i]["stockout_penalty"]
                        * rc.stockout_multiplier
                        for i in range(self.num_products)
                    )
                ),
                "step_fill_rate": step_fill_rate,
                "step_units_sold": units_sold.tolist(),
                "step_units_expired": units_expired.tolist(),
                "step_stockout_units": stockout_units.tolist(),
                "step_demand": demand.tolist(),
                "raw_reward": raw_reward,
            }
        )

        return self._get_obs(), float(reward), done, False, info

    # ------------------------------------------------------------------
    # Step sub-methods
    # ------------------------------------------------------------------
    def _receive_shipments(self):
        """Receive in-transit shipments that arrive today (day index 0)."""
        for i in range(self.num_products):
            arriving = self.in_transit[i, 0]
            if arriving > 0:
                self.inventory[i] += arriving
                if self.product_configs[i]["perishable"]:
                    sl = self.product_configs[i]["shelf_life"]
                    self.expiry_tracker[i].append((arriving, sl))
            # Shift the in-transit pipeline forward
            self.in_transit[i] = np.roll(self.in_transit[i], -1)
            self.in_transit[i, -1] = 0.0

    def _place_orders(self, action: np.ndarray) -> tuple[float, int]:
        """Decode action indices and place orders. Returns (total_cost, emergency_count)."""
        order_costs = 0.0
        emergency_count = 0

        for i in range(self.num_products):
            qty, is_emergency = self.decode_action(int(action[i]), i)

            if qty <= 0:
                continue

            # Check storage capacity
            if self.capacity is not None:
                total_stock = np.sum(self.inventory) + np.sum(self.in_transit)
                available = self.capacity - total_stock
                if available <= 0:
                    continue
                qty = min(qty, int(available))

            # Supplier reliability (partial fulfillment)
            if self.supplier_base_reliability < 1.0 and not is_emergency:
                if self.rng.random() > self.supplier_base_reliability:
                    fill_frac = self.rng.uniform(self.partial_fill_min, 1.0)
                    qty = max(1, int(qty * fill_frac))
                    self.supplier_delivery_history[i].append(fill_frac)
                else:
                    self.supplier_delivery_history[i].append(1.0)
            else:
                self.supplier_delivery_history[i].append(1.0)

            # Keep rolling window at 10
            if len(self.supplier_delivery_history[i]) > 10:
                self.supplier_delivery_history[i].pop(0)

            # Determine lead time and cost
            if is_emergency:
                lt = self.emergency_lead_time
                cost = (
                    self.product_configs[i]["ordering_cost"] * self.emergency_cost_mult
                )
                emergency_count += 1
            else:
                lt = self.rng.integers(self.lead_time_min, self.lead_time_max + 1)
                cost = self.product_configs[i]["ordering_cost"]

            order_costs += cost

            # Place in pipeline (or immediate for emergency with lt=0)
            if lt == 0:
                self.inventory[i] += qty
                if self.product_configs[i]["perishable"]:
                    sl = self.product_configs[i]["shelf_life"]
                    self.expiry_tracker[i].append((float(qty), sl))
            else:
                lt_idx = min(lt - 1, self.max_lead_time - 1)
                self.in_transit[i, lt_idx] += qty

        return order_costs, emergency_count

    def _fulfill_demand(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Generate demand and fulfill from inventory. Returns (demand, sold, stockout, revenue)."""
        demand = self.demand_sim.generate(
            self.day_of_week,
            self.step_count,
            current_inventory=self.inventory.tolist(),
            product_configs=self.product_configs,
        )

        units_sold = np.zeros(self.num_products)
        stockout_units = np.zeros(self.num_products)
        revenue = 0.0

        for i in range(self.num_products):
            d = demand[i]
            sold = min(d, self.inventory[i])
            units_sold[i] = sold
            stockout_units[i] = max(0, d - sold)
            self.inventory[i] -= sold
            revenue += sold * self.product_configs[i]["margin"]

            # Deduct from expiry batches (FIFO — sell oldest first)
            if self.product_configs[i]["perishable"]:
                remaining_to_deduct = sold
                new_batches = []
                for qty_batch, exp in self.expiry_tracker[i]:
                    if remaining_to_deduct <= 0:
                        new_batches.append((qty_batch, exp))
                    elif qty_batch <= remaining_to_deduct:
                        remaining_to_deduct -= qty_batch
                    else:
                        new_batches.append(
                            (qty_batch - remaining_to_deduct, exp)
                        )
                        remaining_to_deduct = 0
                self.expiry_tracker[i] = new_batches

        return demand, units_sold, stockout_units, revenue

    def _expire_perishables(self) -> np.ndarray:
        """Age perishable batches and remove expired stock."""
        units_expired = np.zeros(self.num_products)
        for i in range(self.num_products):
            if not self.product_configs[i]["perishable"]:
                continue
            new_batches = []
            for qty_batch, exp in self.expiry_tracker[i]:
                exp -= 1
                if exp <= 0:
                    units_expired[i] += qty_batch
                    self.inventory[i] = max(0, self.inventory[i] - qty_batch)
                else:
                    new_batches.append((qty_batch, exp))
            self.expiry_tracker[i] = new_batches
            # Update days_to_expiry to min of remaining batches
            if new_batches:
                self.days_to_expiry[i] = min(exp for _, exp in new_batches)
            else:
                self.days_to_expiry[i] = self.product_configs[i]["shelf_life"]
        return units_expired

    def _compute_holding_cost(self) -> float:
        """Sum per-product holding costs."""
        return float(
            sum(
                self.inventory[i] * self.product_configs[i]["holding_cost"]
                for i in range(self.num_products)
            )
        )

    def _compute_reward(
        self,
        revenue: float,
        holding_cost: float,
        order_costs: float,
        units_expired: np.ndarray,
        stockout_units: np.ndarray,
        demand: np.ndarray,
        units_sold: np.ndarray,
    ) -> tuple[float, float]:
        """Compute normalized reward. Returns (reward, raw_reward).

        Components (see RewardConfig for coefficient documentation):
          + revenue               — direct sales income
          − holding_cost × w_h    — penalize excess inventory
          − order_costs × w_o     — penalize over-ordering frequency
          − expiry_penalty        — waste from expired perishables
          − stockout_penalty × w_s — unmet demand signal
          + fill_rate_shaping     — quadratic curve rewarding high fill rate
        """
        rc = self.reward_config

        expiry_penalty = sum(
            units_expired[i] * self.product_configs[i]["expiry_penalty"]
            for i in range(self.num_products)
        )
        stockout_penalty = sum(
            stockout_units[i]
            * self.product_configs[i]["stockout_penalty"]
            * rc.stockout_multiplier
            for i in range(self.num_products)
        )

        # Fill rate for this step
        total_demand_step = demand.sum()
        total_sold_step = units_sold.sum()
        fill_rate = total_sold_step / max(total_demand_step, 1.0)

        # Expected revenue at full fill rate
        expected_revenue = sum(
            self.demand_sim.means[i] * self.product_configs[i]["margin"]
            for i in range(self.num_products)
        )

        # Fill rate shaping: quadratic curve centered at 1.0
        fill_rate_reward = expected_revenue * rc.fill_rate_shape_weight * (
            2.0 * fill_rate - fill_rate * fill_rate - 0.5
        )

        raw_reward = (
            revenue
            - holding_cost * rc.holding_weight
            - order_costs * rc.ordering_weight
            - expiry_penalty
            - stockout_penalty
            + fill_rate_reward
        )

        # Normalize reward to [0, 1] using per-step practical bounds
        max_possible = (
            expected_revenue * 2.0
            + expected_revenue * rc.fill_rate_shape_weight
        )
        min_possible = -(
            sum(
                self.demand_sim.means[i]
                * self.product_configs[i]["stockout_penalty"]
                * rc.stockout_multiplier
                for i in range(self.num_products)
            )
            + expected_revenue * rc.fill_rate_shape_weight
        )
        reward_range = max(max_possible - min_possible, 1.0)
        reward = np.clip(
            (raw_reward - min_possible) / reward_range, SCORE_MIN, SCORE_MAX
        )

        return float(reward), float(raw_reward)

    # ------------------------------------------------------------------
    # Observations & Info
    # ------------------------------------------------------------------
    def _get_obs(self) -> dict[str, Any]:
        storage_frac = (
            np.sum(self.inventory) / self.capacity
            if self.capacity
            else np.sum(self.inventory) / max(np.sum(self.inventory), 1.0)
        )
        return {
            "inventory": self.inventory.astype(np.float32),
            "in_transit": self.in_transit.astype(np.float32),
            "days_to_expiry": self.days_to_expiry.copy(),
            "demand_history": self.demand_history.astype(np.float32),
            "storage_used": np.array([storage_frac], dtype=np.float32),
            "day_of_week": int(self.day_of_week),
            "supplier_reliability": np.array(
                [np.mean(h) for h in self.supplier_delivery_history],
                dtype=np.float32,
            ),
        }

    def _get_info(self) -> dict:
        total_demanded = self.total_units_demanded.sum()
        total_sold = self.total_units_sold.sum()
        fill_rate = total_sold / max(total_demanded, 1.0)
        total_expired = self.total_units_expired.sum()
        waste_rate = total_expired / max(total_sold + total_expired, 1.0)

        # Inventory turnover: uses episode-average inventory (not instantaneous)
        avg_inventory = (
            self._inventory_sum_over_steps / max(self.step_count, 1)
        )
        inventory_turnover = total_sold / max(avg_inventory, 1.0)

        # Service level: fraction of days with fill_rate >= 95%
        service_level = (
            self._high_service_days / max(self.step_count, 1)
        )

        return {
            "fill_rate": fill_rate,
            "waste_rate": waste_rate,
            "service_level": service_level,
            "total_revenue": self.total_revenue,
            "total_holding_cost": self.total_holding_cost,
            "total_ordering_cost": self.total_ordering_cost,
            "total_units_sold": self.total_units_sold.sum(),
            "total_units_expired": total_expired,
            "total_stockout_units": self.total_stockout_units.sum(),
            "total_emergency_orders": self.total_emergency_orders,
            "inventory_turnover": inventory_turnover,
            "step_count": self.step_count,
            "product_names": self.get_product_names(),
        }

    def get_state(self) -> WarehouseState:
        """Return Pydantic WarehouseState model."""
        return WarehouseState(
            inventory=self.inventory.tolist(),
            in_transit=self.in_transit.tolist(),
            days_to_expiry=self.days_to_expiry.tolist(),
            demand_history=self.demand_history.tolist(),
            storage_used=float(
                np.sum(self.inventory) / self.capacity if self.capacity else 0.0
            ),
            day_of_week=int(self.day_of_week),
            supplier_reliability=[
                float(np.mean(h)) for h in self.supplier_delivery_history
            ],
        )
