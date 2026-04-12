"""
Tests for baseline agents.

Validates:
- HeuristicAgent completes episodes and produces valid scores
- FixedQuantityAgent completes episodes and produces valid scores
- Heuristic outperforms fixed-quantity on all tasks
- Agents return correct action dimensions
"""

import numpy as np
import pytest

from baseline.fixed_quantity_agent import FixedQuantityAgent
from baseline.heuristic_agent import HeuristicAgent
from environment.graders import SCORE_MAX, SCORE_MIN, get_grader
from environment.warehouse_env import WarehouseEnv, load_task_config

TASK_IDS = [
    "task1_single_product",
    "task2_multi_product",
    "task3_nonstationary",
]


class TestHeuristicAgent:
    """Verify heuristic agent produces valid episodes."""

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_completes_episode(self, task_id):
        env = WarehouseEnv(task_id=task_id, seed=42)
        agent = HeuristicAgent(env)
        obs, _ = env.reset(seed=42)
        done = False
        steps = 0

        while not done:
            action = agent.act(obs)
            obs, reward, done, _, _ = env.step(action)
            steps += 1

        assert steps == env.max_steps

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_score_in_valid_range(self, task_id):
        env = WarehouseEnv(task_id=task_id, seed=42)
        agent = HeuristicAgent(env)
        config = load_task_config(task_id)
        grader = get_grader(config)

        obs, _ = env.reset(seed=42)
        done = False
        while not done:
            action = agent.act(obs)
            obs, _, done, _, info = env.step(action)

        score = grader.grade(info)
        assert SCORE_MIN <= score <= SCORE_MAX

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_action_dimensions(self, task_id):
        env = WarehouseEnv(task_id=task_id, seed=42)
        agent = HeuristicAgent(env)
        obs, _ = env.reset(seed=42)
        action = agent.act(obs)
        assert len(action) == env.num_products
        assert all(0 <= a < env.actions_per_product for a in action)


class TestFixedQuantityAgent:
    """Verify fixed-quantity agent works correctly."""

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_completes_episode(self, task_id):
        env = WarehouseEnv(task_id=task_id, seed=42)
        agent = FixedQuantityAgent(env, action_index=2)
        obs, _ = env.reset(seed=42)
        done = False
        steps = 0

        while not done:
            action = agent.act(obs)
            obs, _, done, _, _ = env.step(action)
            steps += 1

        assert steps == env.max_steps

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_always_same_action(self, task_id):
        """Fixed-qty agent should return the same action regardless of state."""
        env = WarehouseEnv(task_id=task_id, seed=42)
        agent = FixedQuantityAgent(env, action_index=3)
        obs, _ = env.reset(seed=42)

        action1 = agent.act(obs)
        obs, _, _, _, _ = env.step(action1)
        action2 = agent.act(obs)

        np.testing.assert_array_equal(action1, action2)
        assert all(a == 3 for a in action1)

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_score_in_valid_range(self, task_id):
        env = WarehouseEnv(task_id=task_id, seed=42)
        agent = FixedQuantityAgent(env)
        config = load_task_config(task_id)
        grader = get_grader(config)

        obs, _ = env.reset(seed=42)
        done = False
        while not done:
            action = agent.act(obs)
            obs, _, done, _, info = env.step(action)

        score = grader.grade(info)
        assert SCORE_MIN <= score <= SCORE_MAX


class TestHeuristicBeatsFixed:
    """Heuristic should outperform fixed-quantity on average."""

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_heuristic_outperforms_fixed(self, task_id):
        config = load_task_config(task_id)
        grader = get_grader(config)
        seeds = [42, 43, 44]

        heuristic_scores = []
        fixed_scores = []

        for seed in seeds:
            # Heuristic
            env = WarehouseEnv(task_id=task_id, seed=seed)
            agent = HeuristicAgent(env)
            obs, _ = env.reset(seed=seed)
            done = False
            while not done:
                obs, _, done, _, info = env.step(agent.act(obs))
            heuristic_scores.append(grader.grade(info))

            # Fixed quantity
            env2 = WarehouseEnv(task_id=task_id, seed=seed)
            agent2 = FixedQuantityAgent(env2)
            obs, _ = env2.reset(seed=seed)
            done = False
            while not done:
                obs, _, done, _, info = env2.step(agent2.act(obs))
            fixed_scores.append(grader.grade(info))

        assert np.mean(heuristic_scores) >= np.mean(fixed_scores)
