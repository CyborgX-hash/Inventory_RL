"""
Tests for the FastAPI server endpoints.

Validates:
- /health returns ok
- /tasks returns structured metadata
- /reset initializes the environment correctly
- /step accepts action_ids and returns valid response
- /state returns current state
- /actions returns legal action metadata
- Error handling for invalid inputs
- Response schema completeness
"""

import pytest
from fastapi.testclient import TestClient

from server.app import app


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def reset_env(client):
    """Reset environment before each test that needs it."""
    response = client.post("/reset", json={"task_id": "task1_single_product", "seed": 42})
    assert response.status_code == 200
    return response.json()


class TestHealthEndpoint:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestTasksEndpoint:
    def test_list_tasks(self, client):
        response = client.get("/tasks")
        assert response.status_code == 200
        tasks = response.json()
        assert len(tasks) == 3
        task_ids = [t["id"] for t in tasks]
        assert "task1_single_product" in task_ids
        assert "task2_multi_product" in task_ids
        assert "task3_nonstationary" in task_ids

    def test_task_metadata_fields(self, client):
        response = client.get("/tasks")
        task = response.json()[0]
        assert "id" in task
        assert "difficulty" in task
        assert "max_steps" in task
        assert "num_products" in task
        assert "description" in task
        assert "product_names" in task


class TestResetEndpoint:
    def test_reset_default(self, client):
        response = client.post("/reset")
        assert response.status_code == 200
        data = response.json()
        assert "state" in data
        assert "task_id" in data
        assert "max_steps" in data
        assert "product_names" in data
        assert "legal_actions" in data

    @pytest.mark.parametrize("task_id", [
        "task1_single_product",
        "task2_multi_product",
        "task3_nonstationary",
    ])
    def test_reset_each_task(self, client, task_id):
        response = client.post("/reset", json={"task_id": task_id})
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == task_id

    def test_reset_invalid_task(self, client):
        response = client.post("/reset", json={"task_id": "nonexistent_task"})
        assert response.status_code == 400

    def test_reset_with_seed(self, client):
        response = client.post("/reset", json={"task_id": "task1_single_product", "seed": 123})
        assert response.status_code == 200

    def test_reset_response_schema(self, client):
        """Verify all expected fields are in reset response."""
        response = client.post("/reset", json={"task_id": "task2_multi_product"})
        data = response.json()
        assert data["num_products"] == 3
        assert data["actions_per_product"] == 6
        assert len(data["product_names"]) == 3
        assert len(data["legal_actions"]) == 3


class TestStepEndpoint:
    def test_step_no_init(self, client):
        """Step without reset should work if we reset first."""
        client.post("/reset", json={"task_id": "task1_single_product"})
        response = client.post("/step", json={"action_ids": [0]})
        assert response.status_code == 200

    def test_step_valid(self, client, reset_env):
        response = client.post("/step", json={"action_ids": [0]})
        assert response.status_code == 200
        data = response.json()
        assert "state" in data
        assert "reward" in data
        assert "done" in data
        assert "info" in data
        assert "episode_rewards" in data

    def test_step_wrong_length(self, client, reset_env):
        response = client.post("/step", json={"action_ids": [0, 0, 0]})
        assert response.status_code == 400

    def test_step_out_of_range(self, client, reset_env):
        response = client.post("/step", json={"action_ids": [999]})
        assert response.status_code == 400

    def test_step_negative_action(self, client, reset_env):
        response = client.post("/step", json={"action_ids": [-1]})
        assert response.status_code == 400

    def test_episode_rewards_accumulate(self, client):
        """Episode rewards list should grow with each step."""
        client.post("/reset", json={"task_id": "task1_single_product", "seed": 42})

        for step in range(3):
            response = client.post("/step", json={"action_ids": [0]})
            data = response.json()
            assert len(data["episode_rewards"]) == step + 1

    def test_full_episode(self, client):
        """Run a full episode and verify final score."""
        reset_response = client.post(
            "/reset", json={"task_id": "task1_single_product", "seed": 42}
        )
        max_steps = reset_response.json()["max_steps"]

        for step in range(max_steps):
            response = client.post("/step", json={"action_ids": [0]})
            assert response.status_code == 200
            data = response.json()

            if step == max_steps - 1:
                assert data["done"] is True
                assert data["score"] is not None
                # Score must be in safe range
                assert data["score"] > 0.0
                assert data["score"] < 1.0
                # Episode rewards should have all steps
                assert len(data["episode_rewards"]) == max_steps


class TestStateEndpoint:
    def test_state_after_reset(self, client, reset_env):
        response = client.get("/state")
        assert response.status_code == 200
        state = response.json()
        assert "inventory" in state
        assert "day_of_week" in state


class TestActionsEndpoint:
    def test_actions_after_reset(self, client, reset_env):
        response = client.get("/actions")
        assert response.status_code == 200
        actions = response.json()
        assert len(actions) == 1  # task1 has 1 product
        product = actions[0]
        assert "product_name" in product
        assert "legal_actions" in product
        assert len(product["legal_actions"]) == 6  # 6 ORDER_LEVELS

    def test_actions_task3_emergency(self, client):
        """Task 3 should have 12 actions per product."""
        client.post("/reset", json={"task_id": "task3_nonstationary"})
        response = client.get("/actions")
        actions = response.json()
        assert len(actions) == 5  # 5 products
        for product in actions:
            assert len(product["legal_actions"]) == 12

    def test_actions_schema_completeness(self, client, reset_env):
        """Each action should have all metadata fields."""
        response = client.get("/actions")
        actions = response.json()
        for product in actions:
            for action in product["legal_actions"]:
                assert "index" in action
                assert "order_quantity" in action
                assert "is_emergency" in action
                assert "label" in action
