#!/usr/bin/env python

import os
import argparse
import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from nsfr.utils.common import load_module


def make_env(seed: int, render: bool = False, plusplus: bool = False, noise: bool = False):
    """
    Factory for creating the NudgeEnv used in training.
    """

    env_path = "in/envs/getout/env.py"
    env_module = load_module(env_path)

    def _init():
        env = env_module.NudgeEnv(
            mode="eval",
            plusplus=plusplus,
            noise=noise,
            render=render,
        )
        env.reset(seed=seed)
        return env

    return _init


def evaluate(
    model_path: str,
    episodes: int = 10,
    render: bool = False,
    plusplus: bool = False,
    noise: bool = False,
):

    print(f"Loading best model from: {model_path}")
    model = PPO.load(model_path)

    # Create vectorized eval environment (1 env is enough)
    # env_fn = make_env(seed=np.random.randint(1,10000), render=render, plusplus=plusplus, noise=noise)
    env_fn = make_env(seed=1, render=render, plusplus=plusplus, noise=noise)   
    env = DummyVecEnv([env_fn])

    episode_rewards = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward)

        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}: reward = {total_reward}")

    env.close()

    print("----------------------------------------------------")
    print(f"Average reward over {episodes} episodes: {np.mean(episode_rewards)}")
    print("----------------------------------------------------")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="logs/getout_ppo/best_model/best_model.zip",
                        help="Path to best_model.zip (usually .../best_model/best_model.zip)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--plusplus", action="store_true")
    parser.add_argument("--noise", action="store_true")

    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        episodes=args.episodes,
        render=args.render,
        plusplus=args.plusplus,
        noise=args.noise,
    )


if __name__ == "__main__":
    main()
