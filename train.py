"""
PPO Training Script for the Warehouse Inventory Management Agent.

Uses Stable-Baselines3 with:
  - Normalized observations (VecNormalize) for stable learning
  - Tuned hyperparameters for the multi-product action space
  - Proper episode-length-aligned rollouts
"""

import os
import sys
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor

import torch
import torch.nn as nn

from environment.warehouse_env import WarehouseEnv


# ──────────────────────────────────────────────────────────────
# Gymnasium wrapper: flatten Dict obs + normalize
# ──────────────────────────────────────────────────────────────

class FlattenedWarehouseEnv(gym.Env):
    """Wraps WarehouseEnv to flatten Dict observation into a normalized 1-D Box.

    Normalization scales:
      - inventory / 200   (typical range 0-200)
      - in_transit / 100
      - days_to_expiry / 10  (max shelf life ~5-7 days)
      - demand_history / 100
      - storage_used already in [0,1]
      - day_of_week → one-hot [0,1]
      - supplier_reliability already in [0,1]
    """

    metadata = {"render_modes": []}

    def __init__(self, task_id: str = "task1_single_product", seed: int = 42):
        super().__init__()
        self.wrapped = WarehouseEnv(task_id=task_id, seed=seed)
        self.action_space = self.wrapped.action_space

        # Calculate flattened observation size
        sample_obs = self._flatten_obs(self.wrapped._get_obs())
        self.obs_dim = len(sample_obs)

        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(self.obs_dim,), dtype=np.float32
        )

    def _flatten_obs(self, obs: dict) -> np.ndarray:
        """Flatten and normalize dict observation into a 1-D array."""
        parts = []
        # inventory (N,) — normalize by 200
        parts.append(np.asarray(obs["inventory"], dtype=np.float32).flatten() / 200.0)
        # in_transit (N, max_lt) — normalize by 100
        parts.append(np.asarray(obs["in_transit"], dtype=np.float32).flatten() / 100.0)
        # days_to_expiry (N,) — normalize by 10, -1 → -0.1 for non-perishable
        dte = np.asarray(obs["days_to_expiry"], dtype=np.float32).flatten() / 10.0
        parts.append(dte)
        # demand_history (N, 7) — normalize by 100
        parts.append(np.asarray(obs["demand_history"], dtype=np.float32).flatten() / 100.0)
        # storage_used (1,) — already [0,1]
        parts.append(np.asarray(obs["storage_used"], dtype=np.float32).flatten())
        # day_of_week → one-hot (7,)
        dow = np.zeros(7, dtype=np.float32)
        dow[int(obs["day_of_week"])] = 1.0
        parts.append(dow)
        # supplier_reliability (N,) — already [0,1]
        parts.append(
            np.asarray(obs["supplier_reliability"], dtype=np.float32).flatten()
        )
        return np.concatenate(parts)

    def reset(self, *, seed=None, options=None):
        obs, info = self.wrapped.reset(seed=seed, options=options)
        return self._flatten_obs(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.wrapped.step(action)
        return self._flatten_obs(obs), reward, done, truncated, info


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

def make_env(task_id: str, seed: int):
    """Factory for creating wrapped + monitored environments."""
    def _init():
        env = FlattenedWarehouseEnv(task_id=task_id, seed=seed)
        env = Monitor(env)
        return env
    return _init


class RewardLoggerCallback(BaseCallback):
    """Logs episode reward + fill rate statistics during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        if "infos" in self.locals:
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_count += 1
                    if self.episode_count % 100 == 0:
                        ep_reward = info["episode"]["r"]
                        fill_rate = info.get("fill_rate", 0)
                        print(
                            f"  [Episode {self.episode_count}] "
                            f"Reward: {ep_reward:.3f}, "
                            f"Fill Rate: {fill_rate:.3f}"
                        )
        return True


# Task-specific hyperparameters
TASK_CONFIG = {
    "task1_single_product": {
        "n_steps": 512,     # ~17 episodes per rollout (30-step episodes)
        "batch_size": 64,
        "n_epochs": 10,
        "learning_rate": 3e-4,
        "ent_coef": 0.01,
        "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        "default_timesteps": 100_000,
    },
    "task2_multi_product": {
        "n_steps": 1024,    # ~17 episodes per rollout (60-step episodes)
        "batch_size": 128,
        "n_epochs": 10,
        "learning_rate": 2.5e-4,
        "ent_coef": 0.02,   # More exploration for 3 products
        "net_arch": dict(pi=[128, 128], vf=[128, 128]),
        "default_timesteps": 200_000,
    },
    "task3_nonstationary": {
        "n_steps": 2048,    # ~22 episodes per rollout (90-step episodes)
        "batch_size": 256,
        "n_epochs": 15,
        "learning_rate": 2e-4,
        "ent_coef": 0.03,   # High exploration for 5 products × emergency orders
        "net_arch": dict(pi=[256, 256, 128], vf=[256, 256, 128]),
        "default_timesteps": 500_000,
    },
}


def train_task(
    task_id: str,
    total_timesteps: int = None,
    seed: int = 42,
    save_dir: str = "models",
):
    """Train PPO on a single task with task-specific hyperparameters."""
    cfg = TASK_CONFIG.get(task_id, TASK_CONFIG["task1_single_product"])

    if total_timesteps is None:
        total_timesteps = cfg["default_timesteps"]

    print(f"\n{'='*60}")
    print(f"Training PPO on: {task_id}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"n_steps: {cfg['n_steps']}, batch_size: {cfg['batch_size']}")
    print(f"Network: {cfg['net_arch']}")
    print(f"{'='*60}\n")

    os.makedirs(save_dir, exist_ok=True)

    # Create vectorized environments
    env = DummyVecEnv([make_env(task_id, seed)])
    eval_env = DummyVecEnv([make_env(task_id, seed + 1000)])

    # PPO with task-specific hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lambda progress: cfg["learning_rate"] * (0.3 + 0.7 * progress),
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=0.995,           # High discount — inventory consequences are long-term
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=cfg["ent_coef"],
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=seed,
        policy_kwargs={
            "net_arch": cfg["net_arch"],
            "activation_fn": nn.Tanh,  # Tanh works better with normalized obs
        },
    )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, task_id),
        log_path=os.path.join(save_dir, task_id, "logs"),
        eval_freq=max(total_timesteps // 20, 1000),
        n_eval_episodes=10,
        deterministic=True,
    )

    reward_logger = RewardLoggerCallback()

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, reward_logger],
        progress_bar=True,
    )

    # Save final model
    model_path = os.path.join(save_dir, f"{task_id}_final")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # Evaluation
    print(f"\nPost-training evaluation on {task_id}:")
    evaluate_model(model, task_id, seed=seed + 2000, num_episodes=50)

    env.close()
    eval_env.close()
    return model


def evaluate_model(model, task_id: str, seed: int = 42, num_episodes: int = 50):
    """Evaluate a trained model and print metrics."""
    from environment.graders import get_grader
    from environment.warehouse_env import load_task_config

    config = load_task_config(task_id)
    grader = get_grader(config)

    scores = []
    rewards = []
    fill_rates = []
    waste_rates = []

    for ep in range(num_episodes):
        env = FlattenedWarehouseEnv(task_id=task_id, seed=seed + ep)
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        score = grader.grade(info)
        scores.append(score)
        rewards.append(total_reward)
        fill_rates.append(info.get('fill_rate', 0))
        waste_rates.append(info.get('waste_rate', 0))

    print(f"  Episodes:       {num_episodes}")
    print(f"  Avg Score:      {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"  Avg Reward:     {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
    print(f"  Avg Fill Rate:  {np.mean(fill_rates):.4f} ± {np.std(fill_rates):.4f}")
    print(f"  Avg Waste Rate: {np.mean(waste_rates):.4f} ± {np.std(waste_rates):.4f}")

    return {
        "avg_score": float(np.mean(scores)),
        "avg_reward": float(np.mean(rewards)),
        "avg_fill_rate": float(np.mean(fill_rates)),
        "avg_waste_rate": float(np.mean(waste_rates)),
    }


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train PPO agent for Warehouse Inventory Management"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["task1_single_product", "task2_multi_product", "task3_nonstationary", "all"],
        help="Which task to train on (default: all)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps per task (default: task-specific)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Directory to save models (default: models)",
    )
    args = parser.parse_args()

    tasks = (
        ["task1_single_product", "task2_multi_product", "task3_nonstationary"]
        if args.task == "all"
        else [args.task]
    )

    for task_id in tasks:
        train_task(
            task_id=task_id,
            total_timesteps=args.timesteps,
            seed=args.seed,
            save_dir=args.save_dir,
        )

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
