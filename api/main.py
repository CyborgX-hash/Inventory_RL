"""
FastAPI server for the Warehouse Inventory Management environment.

Endpoints (matching openenv.yaml):
  POST /reset   — Reset environment with a given task_id
  POST /step    — Send an action, receive observation + reward
  GET  /state   — Get current warehouse state
  GET  /tasks   — List available tasks
"""

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

from environment.warehouse_env import WarehouseEnv
from environment.models import OrderAction, StepResult, WarehouseState
from environment.graders import get_grader

app = FastAPI(
    title="Warehouse Inventory Management API",
    description="Multi-product warehouse inventory control under stochastic demand",
    version="1.0.0",
)

# Enable CORS for the frontend UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global environment instance
env: Optional[WarehouseEnv] = None
grader = None


# ---------- Request/Response schemas ----------

class ResetRequest(BaseModel):
    task_id: str = "task1_single_product"
    seed: Optional[int] = None


class ResetResponse(BaseModel):
    state: WarehouseState
    task_id: str
    max_steps: int
    num_products: int


class StepRequest(BaseModel):
    # Support both API standards
    order_quantities: Optional[List[int]] = Field(default=None, alias="action")
    action: Optional[List[int]] = None


class StepResponse(BaseModel):
    state: WarehouseState
    reward: float
    done: bool
    info: Dict[str, Any]
    score: Optional[float] = None  # Final score if done


class TaskInfo(BaseModel):
    id: str
    difficulty: str
    max_steps: int
    num_products: int
    description: str


# ---------- Endpoints ----------

@app.post("/reset", response_model=ResetResponse)
def reset_env(request: Optional[ResetRequest] = None):
    """Reset the environment for a new episode."""
    global env, grader

    # Graceful fallback for empty requests
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
    )


@app.post("/step", response_model=StepResponse)
def step_env(request: Optional[StepRequest] = None):
    """Take one step in the environment."""
    global env, grader

    if env is None:
        raise HTTPException(
            status_code=400, detail="Environment not initialized. Call /reset first."
        )

    # Allow for empty request defaulting to 0 array
    if request is None or (request.order_quantities is None and request.action is None):
        quantities = [0] * env.num_products
    else:
        quantities = request.action if request.action is not None else request.order_quantities

    if len(quantities) != env.num_products:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {env.num_products} order quantities, got {len(quantities)}",
        )

    import numpy as np

    action = np.array(quantities, dtype=np.int64)
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
        score = grader.grade(clean_info)

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


@app.get("/tasks", response_model=List[TaskInfo])
def list_tasks():
    """List all available tasks."""
    from environment.warehouse_env import load_task_config

    tasks = []
    task_ids = [
        "task1_single_product",
        "task2_multi_product",
        "task3_nonstationary",
    ]
    for tid in task_ids:
        cfg = load_task_config(tid)
        tasks.append(
            TaskInfo(
                id=tid,
                difficulty=cfg["difficulty"],
                max_steps=cfg["max_steps"],
                num_products=len(cfg["products"]),
                description=cfg["description"],
            )
        )
    return tasks


@app.get("/health")
def health():
    return {"status": "ok"}
