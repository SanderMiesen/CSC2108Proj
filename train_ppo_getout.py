#!/usr/bin/env python

import os
import argparse
import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Adjust this import path to match your project layout
# This assumes file3 is at: in/envs/getout/env.py
from nsfr.utils.common import load_module
env_path = f"in/envs/getout/env.py"
env_module = load_module(env_path)

EVAL_AND_SAVE = True


def make_env(seed: int, render: bool = False, plusplus: bool = False, noise: bool = False):
    """
    Factory to create NudgeEnv instances for vectorized environments.
    """

    def _init():
        env = env_module.NudgeEnv(
            mode="ppo",
            plusplus=plusplus,
            noise=noise,
            render=render,
        )
        env.reset(seed=seed)
        return env

    return _init


def train(
    model_to_load: str = None,
    random_env: bool = False,
    total_timesteps: int = 1_000_000,
    num_envs: int = 8,
    log_dir: str = "./logs/getout_ppo",
    plusplus: bool = False,
    noise: bool = False,
):
    os.makedirs(log_dir, exist_ok=True)

    # Training env (no rendering)
    env_fns = [make_env(seed=i, render=False, plusplus=plusplus, noise=noise) for i in range(num_envs)]
    env = DummyVecEnv(env_fns)
    env = VecMonitor(env, filename=os.path.join(log_dir, "monitor.csv"))

    # Separate eval env
    if not random_env:
        print("Using fixed seed for eval envs.")
        eval_env_fns = [make_env(seed=1, render=False, plusplus=plusplus, noise=noise) for i in range(2)]
    else:
        print("Using random seeds for eval envs.")
        eval_env_fns = [make_env(seed=np.random.randint(1,10000), render=False, plusplus=plusplus, noise=noise) for _ in range(2)]

    eval_env = DummyVecEnv(eval_env_fns)
    eval_env = VecMonitor(eval_env)

    if model_to_load != "None":
        print(f"Loading model from: {model_to_load}")
        model = PPO.load(model_to_load, env=env)
        model.set_env(env)
    else:
        # PPO hyperparameters – decent starting point
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=1e-3,
            n_steps=2048 // num_envs,  # per env
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            verbose=1,
            tensorboard_log=os.path.join(log_dir, "tb"),
            policy_kwargs=dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]),
        )

    # Checkpoints & evaluation
    checkpoint_callback = CheckpointCallback(
        save_freq=20000 // num_envs,  # in environment steps per env
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="ppo_getout",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=10_000 // num_envs,  # in environment steps per env
        n_eval_episodes=2,
        deterministic=True,
        render=False,
    )

    # without eval:
    if not EVAL_AND_SAVE:
        model.learn(
            total_timesteps=total_timesteps,
            callback=None,
        )
    # with eval:
    else:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
        )

    # Final save
    final_path = os.path.join(log_dir, "ppo_getout_final")
    model.save(final_path)
    print(f"Training finished. Model saved to: {final_path}")

    env.close()
    eval_env.close()


def main(run_nb):
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--log-dir", type=str, default="./logs/getout_ppo/run_" + str(run_nb))
    parser.add_argument("--plusplus", action="store_true", help="Use plusplus mode for NudgeEnv")
    parser.add_argument("--noise", action="store_true", help="Enable noisy logic state")
    parser.add_argument("--model-to-load", type=str, default="logs/getout_ppo/run_110/checkpoints/ppo_getout_1000000_steps.zip")
    # parser.add_argument("--model-to-load", type=str, default="None")
    parser.add_argument("--random-env", type=bool, default = False)
    args = parser.parse_args()

    train(
        model_to_load=args.model_to_load,
        random_env=args.random_env,
        total_timesteps=args.timesteps,
        num_envs=args.num_envs,
        log_dir=args.log_dir,
        plusplus=args.plusplus,
        noise=args.noise,
    )


if __name__ == "__main__":
    run_nb = 110
    main(run_nb)
