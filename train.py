"""
train.py — Train a PPO policy for ArmHandCube-v0 (Franka Panda + Allegro Hand).

Usage:
    python train.py                        # train with defaults (1M steps, 4 envs)
    python train.py --timesteps 5e6        # longer run
    python train.py --n-envs 8             # more parallel envs
    python train.py --eval                 # evaluate a saved policy
    python train.py --eval --model-path checkpoints/best_model

Outputs:
    checkpoints/   — saved policy snapshots
    logs/          — TensorBoard event files
                     (visualize: tensorboard --logdir logs)

Install deps:
    pip install mujoco gymnasium stable-baselines3 tensorboard
"""

import argparse
import os

import gymnasium as gym
import numpy as np

import arm_hand_env  # noqa: F401 — registers ArmHandCube-v0

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

ENV_ID = "ArmHandCube-v0"


def train(args):
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    vec_env  = make_vec_env(ENV_ID, n_envs=args.n_envs)
    eval_env = make_vec_env(ENV_ID, n_envs=1)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        tensorboard_log="./logs/",
        verbose=1,
    )

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path="./checkpoints/",
            log_path="./logs/",
            eval_freq=max(10_000 // args.n_envs, 1),
            n_eval_episodes=10,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=max(50_000 // args.n_envs, 1),
            save_path="./checkpoints/",
            name_prefix="ppo_armhand",
        ),
    ]

    print(f"Training {ENV_ID} for {int(args.timesteps):,} timesteps across {args.n_envs} envs")
    model.learn(
        total_timesteps=int(args.timesteps),
        callback=callbacks,
        tb_log_name="PPO_ArmHand",
    )
    model.save("checkpoints/ppo_armhand_final")
    print("Saved → checkpoints/ppo_armhand_final.zip")


def evaluate(args):
    model_path = args.model_path or "checkpoints/best_model"
    print(f"Loading {model_path}")
    model = PPO.load(model_path)

    env = gym.make(ENV_ID, render_mode="human")
    obs, _ = env.reset()
    ep_reward, ep_rewards = 0.0, []

    for step in range(5000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward

        if info.get("success"):
            print(f"  Step {step}: SUCCESS  err={info['orient_err_deg']:.1f}°")

        if terminated or truncated:
            ep_rewards.append(ep_reward)
            print(f"Episode | reward={ep_reward:.2f}  dropped={info.get('dropped')}  "
                  f"err={info.get('orient_err_deg', 0):.1f}°")
            obs, _ = env.reset()
            ep_reward = 0.0

    env.close()
    if ep_rewards:
        print(f"\nMean episode reward over {len(ep_rewards)} episodes: {np.mean(ep_rewards):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps",  type=float, default=1_000_000)
    parser.add_argument("--n-envs",     type=int,   default=4)
    parser.add_argument("--eval",       action="store_true")
    parser.add_argument("--model-path", type=str,   default=None)
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)
