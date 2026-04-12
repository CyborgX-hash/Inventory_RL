
---
title: Inventory RL
emoji: 📊
colorFrom: blue
colorTo: pink
sdk: docker
pinned: false
---

# 🏭 Adaptive Multi-Product Inventory Management Agent

> **Meta × PyTorch OpenEnv Hackathon 2026 Submission**

A **benchmark-style OpenEnv environment** for multi-product warehouse inventory control under stochastic demand, perishability, supplier unreliability, and emergency procurement. Features a Gymnasium-compatible environment with three difficulty levels, a heuristic baseline, PPO-trained RL agent, and LLM-based inference — all served via a REST API with an interactive React dashboard.

## Why This Matters

Warehouse inventory management is a high-stakes sequential decision problem with real-world impact:
- **$1.8 trillion** in annual inventory distortion costs globally (IHL Group)
- Balancing **stockouts** (lost sales) vs **overstock** (holding costs, waste) under uncertainty
- Perishable goods add a **time-decay dimension** that classical methods struggle with
- Supplier unreliability and demand shocks require **adaptive, real-time decision-making**

This environment captures these challenges in a structured, reproducible benchmark suitable for evaluating RL policies, heuristic baselines, and LLM-based planning agents.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Frontend (React)                       │
│                 Interactive Dashboard                     │
│            localhost:5173 (dev) / :7860 (prod)           │
└───────────────────┬──────────────────────────────────────┘
                    │ HTTP (action_ids)
┌───────────────────▼──────────────────────────────────────┐
│               FastAPI Server (:7860)                      │
│   /reset  /step  /state  /tasks  /actions  /health       │
└───────────────────┬──────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────┐
│            WarehouseEnv (Gymnasium)                       │
│  ┌─────────┐ ┌──────────────┐ ┌────────┐ ┌───────────┐ │
│  │Inventory│ │DemandSimulator│ │Graders │ │  Models   │ │
│  │  Logic  │ │(3 patterns)  │ │(3 tasks)│ │ (Pydantic)│ │
│  └─────────┘ └──────────────┘ └────────┘ └───────────┘ │
└──────────────────────────────────────────────────────────┘
```

## Project Structure

```
inventory_rl/
├── environment/
│   ├── warehouse_env.py      # Gymnasium-compatible warehouse environment
│   ├── demand_simulator.py   # Stochastic demand generation (stationary/seasonal/non-stationary)
│   ├── graders.py            # Task-specific scoring functions with epsilon clamping
│   ├── models.py             # Pydantic data models (State, Action, StepResult, EpisodeSummary)
│   └── __init__.py
├── baseline/
│   └── heuristic_agent.py    # (s, S) reorder-point baseline with reliability compensation
├── server/
│   └── app.py                # FastAPI server (REST API for environment interaction)
├── tasks/
│   ├── task1_easy.yaml       # Single product, stable demand
│   ├── task2_medium.yaml     # 3 products, seasonal demand, storage constraints
│   └── task3_hard.yaml       # 5 products, non-stationary demand, supplier uncertainty
├── tests/
│   ├── test_environment.py   # Core environment correctness tests
│   ├── test_graders.py       # Grader scoring range and edge case tests
│   └── test_api.py           # FastAPI endpoint integration tests
├── frontend/                 # React dashboard (Vite + Lucide icons)
│   ├── src/App.jsx
│   ├── src/index.css
│   └── package.json
├── models/                   # Trained PPO model checkpoints (Git LFS)
├── train.py                  # PPO training script with task-specific hyperparameters
├── inference.py              # Multi-agent evaluation (heuristic / PPO / LLM)
├── benchmark.py              # Multi-seed benchmark comparison script
├── openenv.yaml              # OpenEnv specification
├── pyproject.toml            # Python project config
├── requirements.txt          # Python dependencies
└── Dockerfile                # Multi-stage build (Node + Python)
```

## Three Tasks (Easy → Hard)

| Task | Products | Demand | Lead Time | Storage | Key Challenge |
|------|:---:|--------|:---:|:---:|---------------|
| **Task 1 — Easy** | 1 (non-perishable) | Normal(50, 5) | Fixed: 2 days | Unlimited | Learn basic reorder policy |
| **Task 2 — Medium** | 3 (1 perishable) | Seasonal (2× weekend) | Variable: 1–4 days | 500 units | Balance perishability + shared storage |
| **Task 3 — Hard** | 5 (2 perishable) | Non-stationary + demand shocks | Variable: 1–5 days | 600 units | Supplier reliability 0.7, emergency reorders, cross-product substitution |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `inventory[i]` | float | Current stock level for product i |
| `in_transit[i][d]` | float | Units ordered, arriving in d+1 days |
| `days_to_expiry[i]` | int | Shelf life remaining (-1 = non-perishable) |
| `demand_history[i][7]` | float | Last 7 days of demand per product |
| `storage_used` | float | Fraction of warehouse capacity in use |
| `day_of_week` | int | 0 (Mon) – 6 (Sun) |
| `supplier_reliability[i]` | float | Recent on-time delivery rate (0–1) |

## Action Space (Index-Based)

Actions are **discrete indices**, not raw quantities. Each product gets one action index:

| Index | Order Quantity | Description |
|:---:|:---:|---|
| 0 | 0 units | No order |
| 1 | 5 units | Small order |
| 2 | 10 units | Medium-small |
| 3 | 20 units | Medium |
| 4 | 50 units | Large |
| 5 | 100 units | Very large |

**Task 3 only:** Indices 6–11 are emergency orders of the same quantities (3× cost, instant delivery).

When sending actions via the API, use `action_ids` — an array of indices, one per product:

```json
// Task 1 (1 product): Order 20 units
{"action_ids": [3]}

// Task 2 (3 products): Order 10, 0, 50 units
{"action_ids": [2, 0, 4]}

// Task 3 (5 products, with one emergency order):
{"action_ids": [3, 1, 0, 2, 8]}
//                            ^ emergency 10 units (index 6+2)
```

## Reward Function

```
reward = (units_sold × margin)
       - (inventory_held × holding_cost × 0.3)
       - (order_placed × ordering_cost × 0.5)
       - (units_expired × expiry_penalty)
       - (stockout_units × stockout_penalty × 3.0)
       + fill_rate_shaping_reward
```

Rewards are normalized to the safe interval `(0.001, 0.999)` — never exactly 0 or 1.

## Grading

| Task | Components |
|------|-----------|
| **Task 1** | 70% fill rate + 30% holding cost efficiency |
| **Task 2** | 40% fill rate + 35% waste rate (inverted) + 25% inventory turnover |
| **Task 3** | 40% fill rate + 40% profit vs heuristic baseline + 20% emergency order discipline |

All grader scores are **epsilon-clamped** to `(0.001, 0.999)` to prevent validator rejection.

## Quick Start

### 1. Setup

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/inventory_rl.git
cd inventory_rl
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install frontend dependencies (for development)
cd frontend && npm install && cd ..
```

### 2. Run Heuristic Baseline

```bash
python inference.py --quiet
```

### 3. Train PPO Agent

```bash
# Single task
python train.py --task task1_single_product --timesteps 50000

# All tasks (recommended)
python train.py --task task1_single_product --timesteps 50000
python train.py --task task2_multi_product --timesteps 200000
python train.py --task task3_nonstationary --timesteps 500000
```

### 4. Evaluate Trained Models

```bash
python inference.py --model-dir models --quiet
```

### 5. Run Benchmark

```bash
python benchmark.py --seeds 5
python benchmark.py --model-dir models --seeds 5  # Include PPO comparison
```

### 6. Run Tests

```bash
python -m pytest tests/ -v
```

### 7. Start API Server

```bash
# Backend only
uvicorn server.app:app --port 7860
# → API docs at http://localhost:7860/docs

# Frontend dev server (in separate terminal)
cd frontend && npm run dev
# → Dashboard at http://localhost:5173
```

### 8. Docker

```bash
docker build -t warehouse-inventory .
docker run -p 7860:7860 warehouse-inventory
# → Full app at http://localhost:7860
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Reset environment with `task_id` and optional `seed` |
| `POST` | `/step` | Send `action_ids` (array of indices), receive reward + new state |
| `GET` | `/state` | Get current warehouse state |
| `GET` | `/tasks` | List available tasks with metadata |
| `GET` | `/actions` | Get legal actions with quantity labels per product |
| `GET` | `/health` | Health check |

### API Examples

```bash
# Reset to Task 1
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_single_product", "seed": 42}'

# Take a step (order 20 units)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_ids": [3]}'

# Get legal actions
curl http://localhost:7860/actions

# Get current state
curl http://localhost:7860/state
```

## LLM Agent

The inference script supports LLM-based action selection via the OpenAI API:

```bash
# Using OpenAI directly
export OPENAI_API_KEY="your-key"
python inference.py --use-llm --task task1_single_product

# Using OpenEnv LiteLLM proxy (auto-detected)
export API_BASE_URL="http://proxy:8000/v1"
export API_KEY="your-proxy-key"
python inference.py
```

The LLM agent receives structured prompts with:
- Full product economics (margins, costs, penalties)
- Current inventory levels with days-of-stock analysis
- Legal action indices and their meanings
- Rolling context from previous step's action and reward

On parse failure, it retries up to 2 times before falling back to the heuristic agent.

## Results

### PPO Agent (trained)

| Task | Score | Fill Rate | Waste Rate | Training Steps |
|------|:---:|:---:|:---:|:---:|
| Task 1 (Easy) | 0.93 | **0.999** | 0.000 | 50k |
| Task 2 (Medium) | 0.71 | **0.829** | 0.000 | 200k |
| Task 3 (Hard) | 0.67 | **0.861** | 0.098 | 500k |

### Heuristic Baseline

| Task | Score | Fill Rate | Waste Rate |
|------|:---:|:---:|:---:|
| Task 1 (Easy) | 0.999 | 0.966 | 0.000 |
| Task 2 (Medium) | 0.72 | 0.812 | 0.001 |
| Task 3 (Hard) | 0.63 | 0.616 | 0.003 |

### Key Achievements

- **Task 1**: PPO achieves near-perfect fill rate (0.999) — higher than heuristic (0.966)
- **Task 2**: PPO matches heuristic score while achieving higher fill rate (0.829 vs 0.812)
- **Task 3**: PPO fill rate (0.861) drastically outperforms heuristic (0.616) — a **+40% improvement** in the hardest scenario with unreliable suppliers

## Technical Details

### PPO Configuration

| Parameter | Task 1 | Task 2 | Task 3 |
|-----------|:---:|:---:|:---:|
| n_steps | 512 | 1024 | 2048 |
| batch_size | 64 | 128 | 256 |
| Network | [64, 64] | [128, 128] | [256, 256, 128] |
| ent_coef | 0.01 | 0.02 | 0.03 |
| γ (gamma) | 0.995 | 0.995 | 0.995 |

### Observation Normalization

All observations are manually normalized before being fed to the neural network:
- Inventory / 200, In-transit / 100, Demand history / 100
- Day-of-week → one-hot encoding (7 dims)
- Storage used and supplier reliability already in [0, 1]

## Reproducibility

- All environments accept a `seed` parameter for deterministic evaluation
- Multi-seed benchmarking via `benchmark.py --seeds N`
- Same seed + same actions = identical trajectory (verified in tests)
- Test suite: `python -m pytest tests/ -v`
