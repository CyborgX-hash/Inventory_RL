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
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

from environment.warehouse_env import WarehouseEnv, load_task_config
from environment.models import (
    OrderAction,
    StepResult,
    WarehouseState,
    ProductActionMetadata,
    TaskMetadata,
)
from environment.graders import get_grader, SCORE_MIN, SCORE_MAX

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

# Global environment instance
env: Optional[WarehouseEnv] = None
grader = None


# ---------- Request/Response schemas ----------

class ResetRequest(BaseModel):
    task_id: str = Field(
        default="task1_single_product",
        description="Task to load: task1_single_product, task2_multi_product, or task3_nonstationary",
    )
    seed: Optional[int] = Field(
        default=None, description="Random seed for reproducibility"
    )


class ResetResponse(BaseModel):
    state: WarehouseState
    task_id: str
    max_steps: int
    num_products: int
    product_names: List[str] = Field(
        ..., description="Human-readable names for each product"
    )
    actions_per_product: int = Field(
        ..., description="Number of legal action indices per product"
    )
    legal_actions: List[ProductActionMetadata] = Field(
        ..., description="Structured description of legal actions per product"
    )


class StepRequest(BaseModel):
    action_ids: List[int] = Field(
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
    info: Dict[str, Any] = Field(..., description="Detailed metrics breakdown")
    score: Optional[float] = Field(
        None, description="Final graded score if episode is done (strictly in (0,1))"
    )


# ---------- Endpoints ----------

@app.post("/reset", response_model=ResetResponse)
def reset_env(request: Optional[ResetRequest] = None):
    """Reset the environment for a new episode."""
    global env, grader

    req = request or ResetRequest()

    try:
        env = WarehouseEnv(task_id=req.task_id, seed=req.seed or 42)
        grader = get_grader(env.config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    obs, info = env.reset(seed=req.seed)
    state = env.get_state()

    return ResetResponse(
        state=state,
        task_id=req.task_id,
        max_steps=env.max_steps,
        num_products=env.num_products,
        product_names=env.get_product_names(),
        actions_per_product=env.actions_per_product,
        legal_actions=env.get_legal_actions(),
    )


@app.post("/step", response_model=StepResponse)
def step_env(request: Optional[StepRequest] = None):
    """Take one step in the environment.

    Send action_ids: an array of discrete action indices, one per product.
    Each index maps to a quantity via ORDER_LEVELS = [0, 5, 10, 20, 50, 100].
    """
    global env, grader

    if env is None:
        raise HTTPException(
            status_code=400, detail="Environment not initialized. Call /reset first."
        )

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

    # Convert numpy values in info to Python types
    clean_info = {}
    for k, v in info.items():
        if isinstance(v, (np.floating, np.integer)):
            clean_info[k] = float(v)
        elif isinstance(v, np.ndarray):
            clean_info[k] = v.tolist()
        else:
            clean_info[k] = v

    score = None
    if done and grader is not None:
        raw_score = grader.grade(clean_info)
        # Epsilon clamp is already applied inside grader, but double-check
        score = float(np.clip(raw_score, SCORE_MIN, SCORE_MAX))

    return StepResponse(
        state=state,
        reward=float(reward),
        done=done,
        info=clean_info,
        score=score,
    )


@app.get("/state", response_model=WarehouseState)
def get_state():
    """Get the current warehouse state."""
    global env
    if env is None:
        raise HTTPException(
            status_code=400, detail="Environment not initialized. Call /reset first."
        )
    return env.get_state()


@app.get("/actions", response_model=List[ProductActionMetadata])
def get_actions():
    """Get legal actions with quantity labels for each product."""
    global env
    if env is None:
        raise HTTPException(
            status_code=400, detail="Environment not initialized. Call /reset first."
        )
    return env.get_legal_actions()


@app.get("/tasks", response_model=List[TaskMetadata])
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


# ---------- Static frontend serving ----------

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
