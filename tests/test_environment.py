"""
Tests for the warehouse environment core logic.

Validates:
- Reset output schema and shapes
- Step behavior with valid actions
- Action validation and clamping
- Reward range safety (epsilon-clamped, never exact 0.0 or 1.0)
- Deterministic reproducibility with fixed seeds
- Episode completion
- Metric correctness
- Service level tracking
"""

import numpy as np
import pytest

from environment.warehouse_env import WarehouseEnv, ORDER_LEVELS
from environment.graders import SCORE_MIN, SCORE_MAX


TASK_IDS = [
    "task1_single_product",
    "task2_multi_product",
    "task3_nonstationary",
]


class TestReset:
    """Verify reset produces correct output shapes and types."""

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_reset_returns_obs_and_info(self, task_id):
        env = WarehouseEnv(task_id=task_id, seed=42)
        result = env.reset(seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2
        obs, info = result
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_obs_keys(self, task_id):
        env = WarehouseEnv(task_id=task_id, seed=42)
        obs, _ = env.reset(seed=42)
        expected_keys = {
            "inventory", "in_transit", "days_to_expiry",
            "demand_history", "storage_used", "day_of_week",
            "supplier_reliability",
        }
        assert set(obs.keys()) == expected_keys

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_obs_shapes(self, task_id):
        env = WarehouseEnv(task_id=task_id, seed=42)
        obs, _ = env.reset(seed=42)
        n = env.num_products
        assert obs["inventory"].shape == (n,)
        assert obs["in_transit"].shape == (n, env.max_lead_time)
        assert obs["days_to_expiry"].shape == (n,)
        assert obs["demand_history"].shape == (n, 7)
        assert obs["storage_used"].shape == (1,)
        assert obs["supplier_reliability"].shape == (n,)

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_info_has_expected_fields(self, task_id):
        env = WarehouseEnv(task_id=task_id, seed=42)
        _, info = env.reset(seed=42)
        assert "fill_rate" in info
        assert "product_names" in info
        assert "step_count" in info
        assert "service_level" in info

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_reset_clears_accumulators(self, task_id):
        """After reset, all episode accumulators should be zero."""
        env = WarehouseEnv(task_id=task_id, seed=42)
        # Step once first
        env.reset(seed=42)
        env.step(np.zeros(env.num_products, dtype=np.int64))
        # Now reset
        _, info = env.reset(seed=42)
        assert info["total_revenue"] == 0.0
        assert info["step_count"] == 0


class TestStep:
    """Verify step behavior with valid actions."""

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_step_returns_five_values(self, task_id):
        env = WarehouseEnv(task_id=task_id, seed=42)
        env.reset(seed=42)
        action = np.zeros(env.num_products, dtype=np.int64)
        result = env.step(action)
        assert len(result) == 5
        obs, reward, done, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_reward_in_safe_range(self, task_id):
        """Rewards must always be within [SCORE_MIN, SCORE_MAX]."""
        env = WarehouseEnv(task_id=task_id, seed=42)
        env.reset(seed=42)
        for _ in range(env.max_steps):
            action = np.zeros(env.num_products, dtype=np.int64)
            _, reward, done, _, _ = env.step(action)
            assert SCORE_MIN <= reward <= SCORE_MAX, f"Reward {reward} outside safe range"
            if done:
                break

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_episode_completes(self, task_id):
        """Episode should complete within max_steps."""
        env = WarehouseEnv(task_id=task_id, seed=42)
        env.reset(seed=42)
        done = False
        steps = 0
        while not done:
            action = np.zeros(env.num_products, dtype=np.int64)
            _, _, done, _, _ = env.step(action)
            steps += 1
            assert steps <= env.max_steps + 1, "Episode exceeded max_steps"
        assert steps == env.max_steps

    def test_step_with_max_action(self):
        """Step with maximum action index should not crash."""
        env = WarehouseEnv(task_id="task1_single_product", seed=42)
        env.reset(seed=42)
        max_idx = env.actions_per_product - 1
        action = np.full(env.num_products, max_idx, dtype=np.int64)
        obs, reward, done, truncated, info = env.step(action)
        assert isinstance(reward, float)

    def test_step_info_fields_after_step(self):
        """Info should contain step-level metrics after a step."""
        env = WarehouseEnv(task_id="task1_single_product", seed=42)
        env.reset(seed=42)
        action = np.array([2], dtype=np.int64)  # Order 10 units
        _, _, _, _, info = env.step(action)
        assert "step_revenue" in info
        assert "step_fill_rate" in info
        assert "step_demand" in info
        assert "raw_reward" in info


class TestActionValidation:
    """Verify action validation and clamping."""

    def test_valid_action_passes(self):
        env = WarehouseEnv(task_id="task1_single_product", seed=42)
        action = np.array([3], dtype=np.int64)
        result = env.validate_action(action)
        assert result[0] == 3

    def test_negative_action_clamped(self):
        env = WarehouseEnv(task_id="task1_single_product", seed=42)
        action = np.array([-5], dtype=np.int64)
        result = env.validate_action(action)
        assert result[0] == 0

    def test_overflow_action_clamped(self):
        env = WarehouseEnv(task_id="task1_single_product", seed=42)
        action = np.array([999], dtype=np.int64)
        result = env.validate_action(action)
        assert result[0] == env.actions_per_product - 1

    def test_wrong_length_raises(self):
        env = WarehouseEnv(task_id="task2_multi_product", seed=42)
        action = np.array([0], dtype=np.int64)  # 1 instead of 3
        with pytest.raises(ValueError, match="Expected 3"):
            env.validate_action(action)

    def test_float_action_cast(self):
        """Float actions should be cast to int without error."""
        env = WarehouseEnv(task_id="task1_single_product", seed=42)
        action = np.array([2.7])
        result = env.validate_action(action)
        assert result[0] == 2


class TestDeterminism:
    """Verify deterministic reproducibility with same seed."""

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_same_seed_same_obs(self, task_id):
        env1 = WarehouseEnv(task_id=task_id, seed=123)
        env2 = WarehouseEnv(task_id=task_id, seed=123)
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        np.testing.assert_array_equal(obs1["inventory"], obs2["inventory"])
        np.testing.assert_array_equal(obs1["in_transit"], obs2["in_transit"])

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_same_seed_same_trajectory(self, task_id):
        """Same seed + same actions → identical rewards."""
        rewards1 = _run_episode_zero_action(task_id, seed=99)
        rewards2 = _run_episode_zero_action(task_id, seed=99)
        assert rewards1 == rewards2


class TestLegalActions:
    """Verify legal actions metadata."""

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_legal_actions_per_product(self, task_id):
        env = WarehouseEnv(task_id=task_id, seed=42)
        legal = env.get_legal_actions()
        assert len(legal) == env.num_products
        for pm in legal:
            assert len(pm.legal_actions) == env.actions_per_product
            # Verify indices are sequential
            indices = [a.index for a in pm.legal_actions]
            assert indices == list(range(env.actions_per_product))

    def test_emergency_actions_task3(self):
        env = WarehouseEnv(task_id="task3_nonstationary", seed=42)
        assert env.emergency_enabled
        legal = env.get_legal_actions()
        for pm in legal:
            assert len(pm.legal_actions) == 12
            for a in pm.legal_actions[6:]:
                assert a.is_emergency


class TestDecodeAction:
    """Verify action index → quantity mapping."""

    def test_normal_actions(self):
        env = WarehouseEnv(task_id="task1_single_product", seed=42)
        for idx, expected_qty in enumerate(ORDER_LEVELS):
            qty, is_emergency = env.decode_action(idx, 0)
            assert qty == expected_qty
            assert is_emergency is False

    def test_emergency_actions(self):
        env = WarehouseEnv(task_id="task3_nonstationary", seed=42)
        for idx, expected_qty in enumerate(ORDER_LEVELS):
            qty, is_emergency = env.decode_action(idx + len(ORDER_LEVELS), 0)
            assert qty == expected_qty
            assert is_emergency is True


class TestMetrics:
    """Verify episode metric calculations."""

    def test_fill_rate_zero_when_no_inventory(self):
        """With no ordering, fill rate should drop as inventory depletes."""
        env = WarehouseEnv(task_id="task1_single_product", seed=42)
        env.reset(seed=42)
        done = False
        while not done:
            _, _, done, _, info = env.step(np.array([0], dtype=np.int64))
        # No reordering → fill rate < 1.0 (initial inventory runs out)
        assert info["fill_rate"] < 1.0

    def test_waste_rate_zero_for_non_perishable(self):
        """Task 1 has no perishables → waste_rate should be 0."""
        env = WarehouseEnv(task_id="task1_single_product", seed=42)
        env.reset(seed=42)
        done = False
        while not done:
            _, _, done, _, info = env.step(np.array([3], dtype=np.int64))
        assert info["waste_rate"] == 0.0

    def test_service_level_range(self):
        """Service level should be in [0, 1]."""
        env = WarehouseEnv(task_id="task1_single_product", seed=42)
        env.reset(seed=42)
        done = False
        while not done:
            _, _, done, _, info = env.step(np.array([3], dtype=np.int64))
        assert 0.0 <= info["service_level"] <= 1.0

    def test_inventory_turnover_positive(self):
        """With ordering and sales, turnover should be positive."""
        env = WarehouseEnv(task_id="task1_single_product", seed=42)
        env.reset(seed=42)
        done = False
        while not done:
            _, _, done, _, info = env.step(np.array([3], dtype=np.int64))
        assert info["inventory_turnover"] > 0

    def test_emergency_orders_only_task3(self):
        """Only task3 should have emergency orders tracked."""
        env = WarehouseEnv(task_id="task1_single_product", seed=42)
        env.reset(seed=42)
        done = False
        while not done:
            _, _, done, _, info = env.step(np.array([0], dtype=np.int64))
        assert info["total_emergency_orders"] == 0


class TestWarehouseState:
    """Verify Pydantic state serialization."""

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_get_state_serializable(self, task_id):
        """get_state() should return a JSON-serializable Pydantic model."""
        env = WarehouseEnv(task_id=task_id, seed=42)
        env.reset(seed=42)
        state = env.get_state()
        # Should be serializable to dict
        state_dict = state.model_dump()
        assert isinstance(state_dict, dict)
        assert "inventory" in state_dict
        assert "day_of_week" in state_dict

    def test_state_after_step(self):
        """State should reflect inventory changes after a step."""
        env = WarehouseEnv(task_id="task1_single_product", seed=42)
        env.reset(seed=42)
        state_before = env.get_state()
        env.step(np.array([5], dtype=np.int64))  # Order 100 units
        state_after = env.get_state()
        # Inventory should change due to demand consumption
        assert state_before.inventory != state_after.inventory


# --- Helpers ---

def _run_episode_zero_action(task_id: str, seed: int) -> list:
    """Run a full episode with action=0 for all products."""
    env = WarehouseEnv(task_id=task_id, seed=seed)
    env.reset(seed=seed)
    rewards = []
    done = False
    while not done:
        action = np.zeros(env.num_products, dtype=np.int64)
        _, reward, done, _, _ = env.step(action)
        rewards.append(round(reward, 6))
    return rewards
