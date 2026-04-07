
---
title: Inventory RL
emoji: 📊
colorFrom: blue
colorTo: pink
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Adaptive Multi-Product Inventory Management Agent

An RL agent that manages a warehouse storing multiple perishable and non-perishable products. Every day, it decides how much of each product to reorder — balancing holding costs, ordering costs, stockout penalties, perishability, and stochastic demand. Uses **PPO** (Proximal Policy Optimization) to learn a procurement policy that maximizes profit over a rolling horizon.

## Project Structure

```
inventory_rl/
├── environment/
│   ├── warehouse_env.py      # Gymnasium-compatible warehouse environment
│   ├── demand_simulator.py   # Stochastic demand generation (stationary/seasonal/non-stationary)
│   ├── graders.py            # Task-specific scoring functions
│   └── models.py             # Pydantic data models (State, Action, StepResult)
├── baseline/
│   └── heuristic_agent.py    # (s, S) reorder-point baseline with reliability compensation
├── tasks/
│   ├── task1_easy.yaml       # Single product, stable demand
│   ├── task2_medium.yaml     # 3 products, seasonal demand, storage constraints
│   └── task3_hard.yaml       # 5 products, non-stationary demand, supplier uncertainty
├── api/
│   └── main.py               # FastAPI server (REST API for environment interaction)
├── train.py                  # PPO training script with task-specific hyperparameters
├── inference.py              # Evaluation & logging (heuristic or trained PPO)
├── openenv.yaml              # OpenEnv specification
├── requirements.txt          # Python dependencies
└── Dockerfile                # Container deployment
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
| `in_transit[i][d]` | float | Units ordered, arriving in d days |
| `days_to_expiry[i]` | int | Shelf life remaining (perishables) |
| `demand_history[i][7]` | float | Last 7 days of demand per product |
| `storage_used` | float | % of warehouse capacity in use |
| `day_of_week` | int | 0–6 (demand is seasonal) |
| `supplier_reliability[i]` | float | Recent on-time delivery rate |

## Action Space

For each of N products: `order_quantity[i] ∈ {0, 5, 10, 20, 50, 100}` units

Task 3 also includes emergency reorder options (same quantities at 3× cost, 0-day lead time).

## Reward Function

```
reward = (units_sold × margin)
       - (inventory_held × holding_cost × 0.3)
       - (order_placed × ordering_cost × 0.5)
       - (units_expired × expiry_penalty)
       - (stockout_units × stockout_penalty × 3.0)
       + fill_rate_shaping_reward
```

The fill rate shaping uses a quadratic curve scaled to expected revenue, providing smooth gradient signal for the agent to learn from.

## Quick Start

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Heuristic Baseline

```bash
python inference.py --quiet
```

### Train PPO Agent

```bash
# Train on a specific task
python train.py --task task1_single_product --timesteps 50000

# Train on all tasks (recommended timesteps)
python train.py --task task1_single_product --timesteps 50000
python train.py --task task2_multi_product --timesteps 200000
python train.py --task task3_nonstationary --timesteps 500000
```

### Evaluate Trained Models

```bash
python inference.py --model-dir models --quiet
```

### Start API Server

```bash
uvicorn api.main:app --port 8000
# Interactive docs at http://localhost:8000/docs
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Reset environment with `task_id` |
| `POST` | `/step` | Send order action, get reward + state |
| `GET` | `/state` | Get current warehouse state |
| `GET` | `/tasks` | List available tasks |

### Docker

```bash
docker build -t warehouse-inventory .
docker run -p 8000:8000 warehouse-inventory
```

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
| Task 1 (Easy) | 1.00 | 0.966 | 0.000 |
| Task 2 (Medium) | 0.72 | 0.812 | 0.001 |
| Task 3 (Hard) | 0.63 | 0.616 | 0.003 |

### Key Achievements

- **Task 1**: PPO achieves near-perfect fill rate (0.999) — higher than heuristic (0.966)
- **Task 2**: PPO matches heuristic score while achieving higher fill rate (0.829 vs 0.812)
- **Task 3**: PPO fill rate (0.861) drastically outperforms heuristic (0.616) — a **+40% improvement** in the hardest scenario with unreliable suppliers

## Technical Details

### PPO Configuration

Task-specific hyperparameters are used:

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
- Day-of-week → one-hot encoding
- Storage used and supplier reliability already in [0, 1]

### Grading

- **Task 1**: 70% fill rate score + 30% holding cost efficiency
- **Task 2**: 50% fill rate + 30% waste rate (inverted) + 20% inventory turnover
- **Task 3**: 40% fill rate + 40% profit vs heuristic baseline + 20% emergency order discipline

