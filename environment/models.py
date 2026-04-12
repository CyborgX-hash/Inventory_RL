"""
Typed data models for the Warehouse Inventory Management environment.

Provides Pydantic models for state, actions, step results, and metadata
used across the environment, server API, and inference pipeline.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class WarehouseState(BaseModel):
    """Observable state of the warehouse at a given timestep."""

    inventory: List[float] = Field(
        ..., description="Current stock level per product (units on hand)"
    )
    in_transit: List[List[float]] = Field(
        ..., description="Units in transit per product; in_transit[i][d] = units arriving in d+1 days"
    )
    days_to_expiry: List[int] = Field(
        ..., description="Minimum shelf-life remaining per product (-1 = non-perishable)"
    )
    demand_history: List[List[float]] = Field(
        ..., description="Rolling 7-day demand history per product"
    )
    storage_used: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of warehouse capacity currently occupied"
    )
    day_of_week: int = Field(
        ..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)"
    )
    supplier_reliability: List[float] = Field(
        ..., description="Rolling average on-time delivery rate per product (0.0–1.0)"
    )


class ActionChoice(BaseModel):
    """Describes one legal action for a single product."""

    index: int = Field(..., description="Action index to send in action_ids")
    order_quantity: int = Field(..., description="Number of units this action will order")
    is_emergency: bool = Field(False, description="Whether this is an emergency order (3× cost, instant)")
    label: str = Field(..., description="Human-readable label, e.g. '10 units' or '50 units (emergency)'")


class ProductActionMetadata(BaseModel):
    """Legal actions available for a single product."""

    product_index: int = Field(..., description="Index of this product")
    product_name: str = Field(..., description="Human-readable product name")
    legal_actions: List[ActionChoice] = Field(
        ..., description="All valid actions for this product"
    )


class OrderAction(BaseModel):
    """Action to send to the environment via the /step endpoint.

    Each element is a discrete action INDEX (not a raw quantity).
    Index i maps to ORDER_LEVELS[i] units.
    """

    action_ids: List[int] = Field(
        ..., description=(
            "One action index per product, each indexing into that product's legal_actions. "
            "Index 0 → 0 units, 1 → 5, 2 → 10, 3 → 20, 4 → 50, 5 → 100. "
            "Task 3: indices 6–11 are emergency orders of the same quantities."
        )
    )


class RewardBreakdown(BaseModel):
    """Per-step reward component breakdown."""

    revenue: float = Field(..., description="Revenue from units sold this step")
    holding_cost: float = Field(..., description="Cost of holding inventory this step")
    order_cost: float = Field(..., description="Cost of orders placed this step")
    expiry_penalty: float = Field(..., description="Penalty for expired units this step")
    stockout_penalty: float = Field(..., description="Penalty for unmet demand this step")
    fill_rate: float = Field(..., description="Fraction of demand fulfilled this step")
    raw_reward: float = Field(..., description="Unnormalized reward before scaling")
    normalized_reward: float = Field(..., description="Final reward in (0, 1), epsilon-clamped")


class StepResult(BaseModel):
    """Result returned after each environment step."""

    state: WarehouseState
    reward: float = Field(..., ge=0.0, le=1.0, description="Normalized reward for this step")
    done: bool = Field(..., description="Whether the episode has ended")
    info: Dict[str, Any] = Field(
        ..., description="Detailed metrics: fill_rate, waste_rate, costs, etc."
    )


class TaskMetadata(BaseModel):
    """Metadata describing a single task/difficulty level."""

    id: str = Field(..., description="Unique task identifier")
    difficulty: str = Field(..., description="easy, medium, or hard")
    max_steps: int = Field(..., description="Episode length in days")
    num_products: int = Field(..., description="Number of products to manage")
    description: str = Field(..., description="Human-readable task description")
    product_names: List[str] = Field(..., description="Names of all products")


class EpisodeSummary(BaseModel):
    """Structured summary of a completed episode for benchmarking."""

    task_id: str = Field(..., description="Task that was evaluated")
    agent: str = Field(..., description="Agent type: heuristic, ppo, or llm")
    seed: int = Field(..., description="Random seed used")
    score: float = Field(..., description="Final graded score in (0, 1)")
    total_reward: float = Field(..., description="Sum of per-step rewards")
    fill_rate: float = Field(..., description="Fraction of total demand met")
    waste_rate: float = Field(..., description="Fraction of stock that expired")
    total_revenue: float = Field(..., description="Total revenue from sales")
    total_holding_cost: float = Field(..., description="Total holding cost incurred")
    total_ordering_cost: float = Field(..., description="Total ordering cost incurred")
    total_emergency_orders: int = Field(0, description="Number of emergency orders placed")
    steps: int = Field(..., description="Number of steps completed")