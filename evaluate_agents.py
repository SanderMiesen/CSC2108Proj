import torch
import numpy as np
from pathlib import Path
import os.path as op
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import imageio

from nudge.env import NudgeBaseEnv
# from nudge.utils import make_deterministic
from nudge.agents.logic_agent import LogicPPO
from nudge.agents.neural_agent import NeuralPPO

from nsfr.utils.common import load_module
env_path = f"in/envs/getout/env.py"
env_module = load_module(env_path)

# Helper: load the training configuration
def load_training_config(agent_to_test):
    train_config_path = op.join("evaluate_results", f"agent_{agent_to_test}", "config.yaml")
    with open(train_config_path, "r") as f:
        train_config = yaml.load(f, Loader=yaml.Loader)
    return train_config

# ---------------------------------------------------------
# Helper: build the agent the same way as in training
# ---------------------------------------------------------

def build_agent(env, eval_config, train_config):
    if "logic" in train_config["algorithm"]:
        agent = LogicPPO(
            env=env,
            rules=train_config["rules"],
            lr_actor=train_config["lr_actor"],
            lr_critic=train_config["lr_critic"],
            optimizer=train_config["optimizer"],
            gamma=train_config["gamma"],
            epochs=train_config["epochs"],
            eps_clip=train_config["eps_clip"],
            device=eval_config["device"]
        )
    elif "ppo" in train_config["algorithm"]:
        agent = NeuralPPO(
            env=env,
            lr_actor=train_config["lr_actor"],
            lr_critic=train_config["lr_critic"],
            optimizer=train_config["optimizer"],
            gamma=train_config["gamma"],
            epochs=train_config["epochs"],
            eps_clip=train_config["eps_clip"],
            device=eval_config["device"]
        )
    else:
        raise ValueError(f"Unknown agent type: {train_config["algorithm"]}. Check what the run was on in its config file.")
    return agent

# Helper to make the envs 
def make_env(seed: int, render: bool = False, plusplus: bool = False, noise: bool = False, train_config=None):
    """
    Factory to create NudgeEnv instances for vectorized environments.
    """

    def _init():
        env = env_module.NudgeEnv(
            mode=train_config["algorithm"],
            plusplus=plusplus,
            noise=noise,
            seed=seed, 
            render=render)
        env.reset(seed=seed)
        return env

    return _init()


# ---------------------------------------------------------
# Load agent weights
# ---------------------------------------------------------

def load_checkpoint(agent, checkpoint_path, eval_config):
    print(f"Loading checkpoint from {checkpoint_path} …")
    state_dict = torch.load(checkpoint_path, map_location=eval_config["device"])
    # import ipdb; ipdb.set_trace()
    agent.policy.load_state_dict(state_dict)
    agent.policy_old.load_state_dict(state_dict)
    agent.policy.eval()
    agent.policy_old.eval()


# ---------------------------------------------------------
# Evaluate the agent
# ---------------------------------------------------------

def evaluate(agent, step_nb, eval_config, train_config):
    """
    Evaluate the agent at several checkpoint over N_TEST deterministic environments.
    """
    returns = []

    # Fix environment generation for reproducibility
    rng = np.random.default_rng(eval_config["seed"])

    for i in tqdm(range(eval_config["n_episodes"])):
        # Create environment with deterministic seed
        env_seed = int(rng.integers(0, 100000))
        env = make_env(seed=env_seed, render=eval_config["render"], train_config=train_config)

        state = env.reset(seed=env_seed)
        done = False
        total_reward = 0
        ep_frames = []
        while not done:
            # Step env
            action_pred = agent.select_action(state, epsilon=0.0)
            state, reward, done = env.step(action_pred) 
            total_reward += reward
            # Save frame   
            if eval_config["render"]:
                frame = env.env.render(mode="rgb_array")  # for potential future video recording
                ep_frames.append(frame)
        # Save episode return and end the env
        returns.append(total_reward)
        env.close()
        # Optionally save episode video
        if eval_config["render"]:
            video_dir = op.join("evaluate_results", f"agent_{eval_config["agent_to_test"]}", "videos")
            Path(video_dir).mkdir(parents=True, exist_ok=True)
            video_path = op.join(video_dir, f"step_{step_nb}_episode_{i+1}.mp4")
            with imageio.get_writer(video_path, fps=30) as writer:
                for f in ep_frames:
                    writer.append_data(f)

            
    
    median_return = np.median(returns)
    avg_return = np.mean(returns)
    std_return = np.std(returns)

    print(f"\nEvaluation complete over {eval_config["n_episodes"]} fixed test envs.")
    print(f"Median return:  {median_return:.3f}")
    print(f"Average return: {avg_return:.3f}")
    print(f"Std return:     {std_return:.3f}")

    return returns


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main(eval_config, train_config):
    print("Loading environment template…")
    base_env = NudgeBaseEnv.from_name(name=eval_config["env_name"], mode=train_config["algorithm"])

    print("Building agent…")
    agent = build_agent(base_env, eval_config, train_config)

    steps_to_test = np.arange(eval_config["step_to_start"], eval_config["step_to_end"]+1, eval_config["step_interval"])
    all_returns = {}
    for step_nb in steps_to_test:
        print("Loading checkpoint…")
        checkpoint_path = op.join("evaluate_results", f"agent_{eval_config["agent_to_test"]}", "checkpoints", f"step_{step_nb}.pth")
        load_checkpoint(agent, checkpoint_path, eval_config)

        print("Running evaluation…")
        returns = evaluate(agent, step_nb, eval_config, train_config)
        all_returns[step_nb] = returns
        # Plot returns distribution
        plt.figure(figsize=(8, 5))
        plt.hist(returns, bins=15, color='skyblue', edgecolor='black')
        plt.title(f'Returns Distribution over {eval_config["n_episodes"]} Test Environments\n(Agent: {eval_config["agent_to_test"]}, Step: {step_nb})')
        plt.xlabel('Return')    
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plot_dir = op.join("evaluate_results", f"agent_{eval_config["agent_to_test"]}", "plots")
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(op.join(plot_dir, f'returns_distribution_step_{step_nb}.png'))
        plt.close()
    
    # Save the returns in a CSV file
    df = pd.DataFrame(all_returns)
    csv_dir = op.join("evaluate_results", f"agent_{eval_config["agent_to_test"]}") 
    Path(csv_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(op.join(csv_dir, 'evaluation_returns.csv'), index=False)

if __name__ == "__main__":
    # Load the test config yaml from in/config/default.yaml
    with open(op.join("in", "config", "eval_config.yaml"), 'r') as f:
        eval_config = yaml.safe_load(f)
    # Load the parameters from the training
    train_config = load_training_config(eval_config["agent_to_test"])
    # Little check to make sure user knows what they are doing
    if eval_config["render"] and eval_config["n_episodes"] > 10:
        print("Warning: You are rendering while evaluating over more than 10 episodes. This may slow down evaluation significantly.")
    # Run main
    main(eval_config, train_config)
