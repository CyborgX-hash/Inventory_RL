.PHONY: setup setup-llm test lint format server frontend benchmark train docker clean help

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────

setup:  ## Install core Python dependencies
	pip install -r requirements.txt
	pip install -e ".[dev]"
	@echo "✅ Backend ready. Run 'make setup-frontend' for the dashboard."

setup-frontend:  ## Install frontend dependencies
	cd frontend && npm install

setup-llm:  ## Install optional LLM inference dependencies
	pip install -r requirements-llm.txt

setup-all: setup setup-frontend  ## Install everything

# ──────────────────────────────────────────────
# Quality
# ──────────────────────────────────────────────

test:  ## Run test suite
	python -m pytest tests/ -v --tb=short

test-quick:  ## Run tests without slow markers
	python -m pytest tests/ -v --tb=short -x

lint:  ## Lint Python code with ruff
	python -m ruff check .

format:  ## Format Python code with ruff
	python -m ruff format .

# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────

server:  ## Start FastAPI backend on port 7860
	uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

frontend:  ## Start frontend dev server (proxy to :7860)
	cd frontend && npm run dev

# ──────────────────────────────────────────────
# Training & Evaluation
# ──────────────────────────────────────────────

train:  ## Train PPO agent on all tasks (default timesteps)
	python train.py --task all

train-quick:  ## Quick training run for smoke testing
	python train.py --task task1_single_product --timesteps 5000 --seed 42

benchmark:  ## Run multi-seed benchmark (heuristic only)
	python benchmark.py --seeds 5 --output results/benchmark.json

benchmark-ppo:  ## Benchmark with PPO comparison
	python benchmark.py --seeds 5 --model-dir models --output results/benchmark_ppo.json

inference:  ## Run heuristic inference on all tasks
	python inference.py --quiet

# ──────────────────────────────────────────────
# Docker
# ──────────────────────────────────────────────

docker:  ## Build Docker image
	docker build -t warehouse-inventory .

docker-run:  ## Run Docker container
	docker run -p 7860:7860 warehouse-inventory

# ──────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────

clean:  ## Remove generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build dist htmlcov .coverage
	@echo "✅ Cleaned."

# ──────────────────────────────────────────────
# Help
# ──────────────────────────────────────────────

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
