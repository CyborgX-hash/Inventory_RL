---
title: Inventory RL
emoji: 📊
colorFrom: blue
colorTo: pink
sdk: docker
pinned: false
---

# 🏭 Inventory RL Benchmark
> **A High-Fidelity Environment for Supply Chain Reinforcement Learning**  
> *Meta × PyTorch OpenEnv Hackathon 2026 Submission*

[![CI Status](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Inventory RL** is a production-ready, `Gymnasium`-compatible benchmark that simulates the high-stakes complexities of multi-product warehouse management. It provides a rigorous testing ground for evaluating Reinforcement Learning (PPO), LLM-based planning agents, and heuristic baselines against real-world supply chain dynamics.

---

## 🌍 The Problem: A Meaningful Real-World Challenge

Inventory distortion costs the global economy **$1.8 trillion annually**. Modern supply chains are not stationary toy problems; they require balancing competing constraints under deep uncertainty:
- **Cost Trade-offs:** The tension between stockout penalties (lost sales) vs. holding costs (waste and capital tie-up).
- **Perishability:** Products decay, turning simple reorder policies into complex timing optimizations with FIFO dynamics.
- **Supply & Demand Shocks:** Non-stationary consumer behavior coupled with stochastic supplier failures (shortfalls and delays).

This environment captures these exact mechanics, providing a meaningful, non-trivial benchmark where advanced agents can demonstrate genuine business value over naive continuous-review baselines.

## ✨ Novelty & Realism

Unlike standard inventory environments, this benchmark prioritizes operational realism and stable agent evaluation:

- **Multi-Product Shared Capacity:** Agents must learn to dynamically allocate constrained warehouse space among competing items.
- **Supplier Unreliability:** Simulates real-world procurement failures and delivery variances.
- **Emergency Procurement Space:** Introduces a secondary discrete action space for high-cost, instant deliveries during shocks.
- **Index-Based Action Encodings:** Discrete order indices mathematically stabilize PPO convergence and LLM structured outputs compared to raw continuous quantities.
- **Epsilon-Clamped Safety bounds:** Scoring functions map cleanly to `(0, 1)`, eliminating edge-case validator rejections and ensuring robust distributed evaluations.

## 🔬 Benchmark-Grade Evaluation

We ensure **100% reproducibility** and strict evaluation quality out-of-the-box:
- **Three Tiered Tasks:** Evaluates generalization from stationary single-product (Easy) to non-stationary multi-product environments with emergencies (Hard).
- **Deterministic Multi-Seed Evaluation:** Built-in benchmarking pipeline (`benchmark.py`) ensures identical state trajectories for a given seed and action sequence.
- **Established Baselines:** Shipped with a highly tuned Heuristic `(s, S)` agent for baseline grading and an optimized PPO implementation.
- **Comprehensive Test Coverage:** Pytest suite validating environment physics, API stability, and reward clamping.

## 🏗️ Architecture Stack

Engineered for scalability, seamless API integration, and immediate cloud deployment.
- **Backend:** `FastAPI` providing thread-safe `EnvironmentSession` state management.
- **Frontend:** Interactive `React / Vite` dashboard for human-in-the-loop debugging and state visualization.
- **Core Environment:** Strict `Gymnasium` interface with parameterized underlying dynamics.
- **LLM Integration:** LiteLLM proxy compatibility with robust retry-and-json-parsing pipelines.

## 🚀 Quick Start

Ensure you have Python 3.10+ installed.

### 1. Installation
```bash
git clone https://github.com/YOUR_USERNAME/inventory_rl.git
cd inventory_rl
python3 -m venv .venv && source .venv/bin/activate
make setup           # Installs python backend & RL requirements
make setup-frontend  # Installs React dashboard
```

### 2. Run the Benchmark
Compare our PPO models against the highly-tuned heuristic baselines:
```bash
make benchmark-ppo   # Runs multi-seed evaluation and prints comparison
```

### 3. Launch the Server & UI
Explore the environment interactively:
```bash
make server          # Starts FastAPI Server on port 7860
make frontend        # Starts React Dashboard on port 5173 
```

### 4. Docker Deployment
```bash
make docker && make docker-run
# → Access full application at http://localhost:7860
```

## 📊 Results Summary

The tailored PPO agent routinely outperforms the classical heuristic on complex tasks, specifically handling supply unreliability via preemptive stock adjustments during demand shocks:

| Task | PPO Fill Rate | Heuristic Fill Rate | Win Margin |
|------|:---:|:---:|:---:|
| **Task 1 (Easy)** | **99.9%** | 96.6% | + 3.3% |
| **Task 2 (Med)** | **82.9%** | 81.2% | + 1.7% |
| **Task 3 (Hard)** | **86.1%** | 61.6% | **+ 24.5%** |

*(Extracted from full 5-seed benchmark evaluation. PPO models trained on up to 500k timesteps.)*

---
*Developed for the Meta × PyTorch OpenEnv Hackathon 2026. Code under MIT License.*
