"""
Smoke tests for the benchmark pipeline.

Validates:
- Benchmark evaluation functions complete without errors
- Results have expected structure
- Scores are in valid range
"""

import numpy as np
import pytest

from environment.graders import SCORE_MIN, SCORE_MAX


class TestBenchmarkSmoke:
    """Verify benchmark pipeline runs without crashing."""

    def test_heuristic_evaluator(self):
        """Heuristic evaluation should produce valid results."""
        from benchmark import evaluate_heuristic

        result = evaluate_heuristic("task1_single_product", seeds=[42])
        assert result["agent"] == "heuristic"
        assert "avg_score" in result
        assert "avg_fill_rate" in result
        assert "avg_profit" in result
        assert SCORE_MIN <= result["avg_score"] <= SCORE_MAX

    def test_fixed_qty_evaluator(self):
        """Fixed-quantity evaluation should produce valid results."""
        from benchmark import evaluate_fixed_quantity

        result = evaluate_fixed_quantity("task1_single_product", seeds=[42])
        assert result["agent"] == "fixed_qty"
        assert SCORE_MIN <= result["avg_score"] <= SCORE_MAX

    def test_multi_seed_consistency(self):
        """Multiple seeds should produce consistent (but not identical) results."""
        from benchmark import evaluate_heuristic

        result = evaluate_heuristic("task1_single_product", seeds=[42, 43, 44])
        assert result["num_seeds"] == 3
        assert len(result["scores"]) == 3
        # Scores should be close but not identical (different seeds)
        assert result["std_score"] < 0.5  # Not wildly different

    def test_all_tasks_complete(self):
        """All three tasks should produce results."""
        from benchmark import evaluate_heuristic

        for task_id in [
            "task1_single_product",
            "task2_multi_product",
            "task3_nonstationary",
        ]:
            result = evaluate_heuristic(task_id, seeds=[42])
            assert result["task_id"] == task_id
            assert SCORE_MIN <= result["avg_score"] <= SCORE_MAX
