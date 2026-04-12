"""
FastAPI server for the Warehouse Inventory Management environment.

Endpoints (matching openenv.yaml):
  POST /reset    — Reset environment with a given task_id
  POST /step     — Send action_ids, receive observation + reward
  GET  /state    — Get current warehouse state
  GET  /tasks    — List available tasks
  GET  /actions  — Get legal actions with quantity labels
  GET  /health   — Health check
"""

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from environment.graders import SCORE_MAX, SCORE_MIN, get_grader
from environment.models import (
    ProductActionMetadata,
    TaskMetadata,
    WarehouseState,
)
from environment.warehouse_env import WarehouseEnv, load_task_config

app = FastAPI(
    title="Warehouse Inventory Management API",
    description=(
        "Multi-product warehouse inventory control under stochastic demand, "
        "perishability, and supplier unreliability. "
        "Actions are discrete indices into ORDER_LEVELS = [0, 5, 10, 20, 50, 100]."
    ),
    version="1.0.0",
)

# Enable CORS for the frontend UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────────────────────────
# Session Manager — replaces fragile global mutable state
# ────────────────────────────────────────────────────────────────


@dataclass
class EnvironmentSession:
    """Encapsulates a single environment episode.

    Groups the environment instance, grader, and episode metadata
    into one object so the server never has dangling global refs.
    """

    env: WarehouseEnv
    grader: Any
    task_id: str
    episode_rewards: list = field(default_factory=list)

    @classmethod
    def create(cls, task_id: str, seed: int = 42) -> "EnvironmentSession":
        """Factory: build a fresh session from a task_id."""
        env = WarehouseEnv(task_id=task_id, seed=seed)
        config = load_task_config(task_id)
        grader = get_grader(config)
        return cls(env=env, grader=grader, task_id=task_id)


# Active session (one at a time — fine for single-user hackathon demo)
_session: EnvironmentSession | None = None


def _get_session() -> EnvironmentSession:
    """Return the active session or raise an HTTP 400 error."""
    if _session is None:
        raise HTTPException(
            status_code=400, detail="Environment not initialized. Call /reset first."
        )
    return _session


# ────────────────────────────────────────────────────────────────
# Request / Response schemas
# ────────────────────────────────────────────────────────────────


class ResetRequest(BaseModel):
    task_id: str = Field(
        default="task1_single_product",
        description="Task to load: task1_single_product, task2_multi_product, or task3_nonstationary",
    )
    seed: int | None = Field(default=None, description="Random seed for reproducibility")


class ResetResponse(BaseModel):
    state: WarehouseState
    task_id: str
    max_steps: int
    num_products: int
    product_names: list[str] = Field(..., description="Human-readable names for each product")
    actions_per_product: int = Field(
        ..., description="Number of legal action indices per product"
    )
    legal_actions: list[ProductActionMetadata] = Field(
        ..., description="Structured description of legal actions per product"
    )


class StepRequest(BaseModel):
    action_ids: list[int] = Field(
        ...,
        description=(
            "One action index per product. Each index selects from ORDER_LEVELS: "
            "0→no order, 1→5 units, 2→10 units, 3→20 units, 4→50 units, 5→100 units. "
            "For Task 3, indices 6–11 are emergency orders of the same quantities."
        ),
    )


class StepResponse(BaseModel):
    state: WarehouseState
    reward: float = Field(..., description="Normalized reward for this step (0–1)")
    done: bool = Field(..., description="Whether the episode has ended")
    info: dict[str, Any] = Field(..., description="Detailed metrics breakdown")
    score: float | None = Field(
        None, description="Final graded score if episode is done (strictly in (0,1))"
    )
    episode_rewards: list[float] = Field(
        default_factory=list, description="Cumulative reward history for charting"
    )


# ────────────────────────────────────────────────────────────────
# Endpoints
# ────────────────────────────────────────────────────────────────


@app.post("/reset", response_model=ResetResponse)
def reset_env(request: ResetRequest | None = None):
    """Reset the environment for a new episode."""
    global _session

    req = request or ResetRequest()

    try:
        _session = EnvironmentSession.create(task_id=req.task_id, seed=req.seed or 42)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    _session.env.reset(seed=req.seed)
    state = _session.env.get_state()

    return ResetResponse(
        state=state,
        task_id=req.task_id,
        max_steps=_session.env.max_steps,
        num_products=_session.env.num_products,
        product_names=_session.env.get_product_names(),
        actions_per_product=_session.env.actions_per_product,
        legal_actions=_session.env.get_legal_actions(),
    )


@app.post("/step", response_model=StepResponse)
def step_env(request: StepRequest | None = None):
    """Take one step in the environment.

    Send action_ids: an array of discrete action indices, one per product.
    Each index maps to a quantity via ORDER_LEVELS = [0, 5, 10, 20, 50, 100].
    """
    session = _get_session()
    env = session.env

    # Default to "no order" for all products if request is empty
    if request is None or request.action_ids is None:
        action_ids = [0] * env.num_products
    else:
        action_ids = request.action_ids

    if len(action_ids) != env.num_products:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {env.num_products} action_ids, got {len(action_ids)}",
        )

    # Validate index range
    for i, aid in enumerate(action_ids):
        if aid < 0 or aid >= env.actions_per_product:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"action_ids[{i}]={aid} is out of range. "
                    f"Valid range: 0–{env.actions_per_product - 1}"
                ),
            )

    action = np.array(action_ids, dtype=np.int64)
    obs, reward, done, truncated, info = env.step(action)
    state = env.get_state()

    # Track episode reward history
    session.episode_rewards.append(float(reward))

    # Convert numpy values in info to Python types
    clean_info = _serialize_info(info)

    score = None
    if done and session.grader is not None:
        raw_score = session.grader.grade(clean_info)
        score = float(np.clip(raw_score, SCORE_MIN, SCORE_MAX))

    return StepResponse(
        state=state,
        reward=float(reward),
        done=done,
        info=clean_info,
        score=score,
        episode_rewards=session.episode_rewards,
    )


@app.get("/state", response_model=WarehouseState)
def get_state():
    """Get the current warehouse state."""
    return _get_session().env.get_state()


@app.get("/actions", response_model=list[ProductActionMetadata])
def get_actions():
    """Get legal actions with quantity labels for each product."""
    return _get_session().env.get_legal_actions()


@app.get("/tasks", response_model=list[TaskMetadata])
def list_tasks():
    """List all available tasks with metadata."""
    tasks = []
    task_ids = [
        "task1_single_product",
        "task2_multi_product",
        "task3_nonstationary",
    ]
    for tid in task_ids:
        cfg = load_task_config(tid)
        tasks.append(
            TaskMetadata(
                id=tid,
                difficulty=cfg["difficulty"],
                max_steps=cfg["max_steps"],
                num_products=len(cfg["products"]),
                description=cfg["description"],
                product_names=[p["name"] for p in cfg["products"]],
            )
        )
    return tasks


@app.get("/health")
def health():
    return {"status": "ok"}


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────


def _serialize_info(info: dict) -> dict:
    """Convert numpy types in info dict to JSON-safe Python types."""
    clean = {}
    for k, v in info.items():
        if isinstance(v, (np.floating, np.integer)):
            clean[k] = float(v)
        elif isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        else:
            clean[k] = v
    return clean


# ────────────────────────────────────────────────────────────────
# Static frontend serving
# ────────────────────────────────────────────────────────────────

static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")
if os.path.exists(static_dir):
    assets_dir = os.path.join(static_dir, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        """Serve React frontend for root or unknown paths."""
        if full_path == "" or not os.path.exists(os.path.join(static_dir, full_path)):
            return FileResponse(os.path.join(static_dir, "index.html"))
        return FileResponse(os.path.join(static_dir, full_path))


def main():
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
