#!/usr/bin/env python

import os
import os.path as op
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from nsfr.utils.common import load_module

TEST_SEED = 9999        # ensures reproducibility
RENDER = False         # whether to render the environment during evaluation
ENV_NAME = "getout"
N_TEST = 50 # number of test episodes


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
            seed=seed,
        )
        env.reset(seed=seed)
        return env

    return _init


def evaluate(
    model_path: str,
    render: bool = RENDER
):
    """
    Evaluate a trained PPO agent over a number of episodes.
    """

    episode_rewards = [] 
    # Fix environment generation for reproducibility
    rng = np.random.default_rng(TEST_SEED)
    # Load trained model
    model = PPO.load(model_path)

    for i in tqdm(range(N_TEST)):
        env_seed = int(rng.integers(0, 100000))
        env_fn = make_env(seed=env_seed, render=render)
        env = DummyVecEnv([env_fn])

        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward)
        if not env.envs[0].env.has_started:
            print("wtf")
        episode_rewards.append(total_reward)
        env.close()

    median_return = np.median(episode_rewards)
    avg_return = np.mean(episode_rewards)
    std_return = np.std(episode_rewards)

    print(f"\nEvaluation complete over {N_TEST} fixed test envs.")
    print(f"Median return:  {median_return:.3f}")
    print(f"Average return: {avg_return:.3f}")
    print(f"Std return:     {std_return:.3f}")

    return episode_rewards


def main(agent_to_test: str, steps_to_test: np.ndarray):
    model_path = op.join("evaluate_results", f"agent_{agent_to_test}")

    all_returns = {}
    for step_nb in steps_to_test:
        # Compute model path
        base_model_paths = op.join("evaluate_results", f"agent_{agent_to_test}")
        if step_nb <= 1000000:
            model_path = op.join(base_model_paths, "checkpoints_1", f"ppo_getout_{step_nb}_steps")
        else:
            model_path = op.join(base_model_paths, "checkpoints_2", f"ppo_getout_{step_nb-1000000}_steps")
        # Evaluate the agent
        returns = evaluate(
            model_path=model_path,
            render=RENDER
        )
        all_returns[step_nb] = returns
        # Plot returns distribution
        plt.figure(figsize=(8, 5))
        print(returns)
        plt.hist(returns, bins=15, color='skyblue', edgecolor='black')
        plt.title(f'Returns Distribution over {N_TEST} Test Environments\n(Agent: {agent_to_test}, Step: {step_nb})')
        plt.xlabel('Return')    
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plot_dir = op.join("evaluate_results", f"agent_{agent_to_test}", "plots")
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(op.join(plot_dir, f'returns_distribution_step_{step_nb}.png'))
        plt.close()
    
    # Save the returns in a CSV file
    df = pd.DataFrame(all_returns)
    csv_dir = op.join("evaluate_results", f"agent_{agent_to_test}") 
    Path(csv_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(op.join(csv_dir, 'evaluation_returns.csv'), index=False)

if __name__ == "__main__":
    agent_to_test = "ppo_custom"
    step_size = 100000
    steps_to_test = np.arange(step_size, 1810000, step_size)
    main(agent_to_test, steps_to_test)
