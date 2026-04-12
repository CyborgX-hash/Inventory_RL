"""
Tests for grader correctness.

Validates:
- All graders produce scores in the safe (SCORE_MIN, SCORE_MAX) range
- Edge cases: perfect performance, zero performance, high waste
- Score monotonicity: better fill rate → higher score
"""

import numpy as np
import pytest

from environment.graders import (
    SCORE_MAX,
    SCORE_MIN,
    Task1Grader,
    Task2Grader,
    Task3Grader,
    _safe_score,
    get_grader,
)
from environment.warehouse_env import WarehouseEnv, load_task_config


class TestSafeScore:
    """Verify epsilon clamping utility."""

    def test_clamps_zero(self):
        assert _safe_score(0.0) == SCORE_MIN

    def test_clamps_one(self):
        assert _safe_score(1.0) == SCORE_MAX

    def test_clamps_negative(self):
        assert _safe_score(-0.5) == SCORE_MIN

    def test_clamps_above_one(self):
        assert _safe_score(1.5) == SCORE_MAX

    def test_preserves_middle(self):
        assert _safe_score(0.5) == 0.5

    def test_never_exact_zero(self):
        assert _safe_score(0.0) > 0.0

    def test_never_exact_one(self):
        assert _safe_score(1.0) < 1.0


class TestTask1Grader:
    """Task 1: fill rate + holding cost."""

    def setup_method(self):
        config = load_task_config("task1_single_product")
        self.grader = get_grader(config)

    def test_perfect_performance(self):
        info = {
            "fill_rate": 1.0,
            "total_holding_cost": 0.0,
            "step_count": 30,
        }
        score = self.grader.grade(info)
        assert SCORE_MIN <= score <= SCORE_MAX

    def test_zero_fill_rate(self):
        info = {
            "fill_rate": 0.0,
            "total_holding_cost": 0.0,
            "step_count": 30,
        }
        score = self.grader.grade(info)
        assert SCORE_MIN <= score <= SCORE_MAX

    def test_high_holding_cost(self):
        info = {
            "fill_rate": 0.95,
            "total_holding_cost": 5000.0,
            "step_count": 30,
        }
        score = self.grader.grade(info)
        assert SCORE_MIN <= score <= SCORE_MAX

    def test_better_fill_rate_higher_score(self):
        info_low = {"fill_rate": 0.3, "total_holding_cost": 100, "step_count": 30}
        info_high = {"fill_rate": 0.9, "total_holding_cost": 100, "step_count": 30}
        assert self.grader.grade(info_low) < self.grader.grade(info_high)


class TestTask2Grader:
    """Task 2: fill rate + waste rate + turnover."""

    def setup_method(self):
        config = load_task_config("task2_multi_product")
        self.grader = get_grader(config)

    def test_score_range(self):
        info = {
            "fill_rate": 0.8,
            "waste_rate": 0.1,
            "inventory_turnover": 4.0,
        }
        score = self.grader.grade(info)
        assert SCORE_MIN <= score <= SCORE_MAX

    def test_no_waste_bonus(self):
        info_waste = {"fill_rate": 0.8, "waste_rate": 0.4, "inventory_turnover": 4.0}
        info_clean = {"fill_rate": 0.8, "waste_rate": 0.0, "inventory_turnover": 4.0}
        assert self.grader.grade(info_clean) > self.grader.grade(info_waste)


class TestTask3Grader:
    """Task 3: fill rate + profit + emergency discipline."""

    def setup_method(self):
        config = load_task_config("task3_nonstationary")
        self.grader = get_grader(config)
        # Use a fixed baseline for test determinism
        self.grader.set_baseline_profit(50000.0)

    def test_score_range(self):
        info = {
            "fill_rate": 0.7,
            "total_revenue": 100000.0,
            "total_holding_cost": 20000.0,
            "total_ordering_cost": 10000.0,
            "total_units_expired": 50.0,
            "total_emergency_orders": 3,
        }
        score = self.grader.grade(info)
        assert SCORE_MIN <= score <= SCORE_MAX

    def test_emergency_overuse_penalty(self):
        base_info = {
            "fill_rate": 0.8,
            "total_revenue": 100000.0,
            "total_holding_cost": 20000.0,
            "total_ordering_cost": 10000.0,
            "total_units_expired": 0.0,
        }
        info_low_emerg = {**base_info, "total_emergency_orders": 2}
        info_high_emerg = {**base_info, "total_emergency_orders": 20}
        assert self.grader.grade(info_low_emerg) > self.grader.grade(info_high_emerg)


class TestGraderFactory:
    """Verify get_grader returns correct type."""

    @pytest.mark.parametrize("task_id,expected_type", [
        ("task1_single_product", Task1Grader),
        ("task2_multi_product", Task2Grader),
        ("task3_nonstationary", Task3Grader),
    ])
    def test_factory(self, task_id, expected_type):
        config = load_task_config(task_id)
        grader = get_grader(config)
        assert isinstance(grader, expected_type)


class TestEndToEndGrading:
    """Run a full episode and verify the grader produces a valid score."""

    @pytest.mark.parametrize("task_id", [
        "task1_single_product",
        "task2_multi_product",
        "task3_nonstationary",
    ])
    def test_full_episode_score(self, task_id):
        env = WarehouseEnv(task_id=task_id, seed=42)
        config = load_task_config(task_id)
        grader = get_grader(config)
        obs, info = env.reset(seed=42)

        done = False
        while not done:
            # Use action=0 (no order) for all products
            action = np.zeros(env.num_products, dtype=np.int64)
            obs, reward, done, truncated, info = env.step(action)

        score = grader.grade(info)
        assert SCORE_MIN <= score <= SCORE_MAX
        assert score > 0.0, "Score should never be exactly 0.0"
        assert score < 1.0, "Score should never be exactly 1.0"
