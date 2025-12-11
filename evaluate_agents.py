import torch
import numpy as np
from pathlib import Path
import os.path as op
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from nudge.env import NudgeBaseEnv
# from nudge.utils import make_deterministic
from nudge.agents.logic_agent import LogicPPO
from nudge.agents.neural_agent import NeuralPPO

from nsfr.utils.common import load_module
env_path = f"in/envs/getout/env.py"
env_module = load_module(env_path)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Hyperparameters MUST match the ones used to TRAIN the agent
RULES = "getout_bs_top10"
LR_ACTOR = 0.001
LR_CRITIC = 0.0003
GAMMA = 0.99
EPOCHS = 20
EPS_CLIP = 0.2
OPTIMIZER = torch.optim.Adam

# Helper: load the training configuration
def load_training_config(agent_to_test):
    train_config_path = op.join("evaluate_results", f"agent_{agent_to_test}", "config.yaml")
    with open(train_config_path, "r") as f:
        train_config = yaml.load(f, Loader=yaml.Loader)
    return train_config

# ---------------------------------------------------------
# Helper: build the agent the same way as in training
# ---------------------------------------------------------

def build_agent(env, agent_to_test, eval_config, train_config):
    if "logic" in agent_to_test:
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
    elif agent_to_test == "ppo_paper":
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
            seed=seed)
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

def evaluate(agent, eval_config, train_config):
    """
    Evaluate the agent at several checkpoint over N_TEST deterministic environments.
    """
    returns = []

    # Fix environment generation for reproducibility
    rng = np.random.default_rng(eval_config["seed"])

    for i in tqdm(range(eval_config["n_episodes"])):
        # Create environment with deterministic seed
        env_seed = int(rng.integers(0, 100000))
        env = make_env(seed=env_seed, render=False, train_config=train_config)

        state = env.reset(seed=env_seed)
        done = False
        total_reward = 0

        while not done:
            action_pred = agent.select_action(state, epsilon=0.0)
            state, reward, done = env.step(action_pred)
            total_reward += reward

        returns.append(total_reward)
        env.close()
    
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
    base_env = NudgeBaseEnv.from_name(eval_config["env_name"], mode=train_config["algorithm"])

    print("Building agent…")
    agent = build_agent(base_env, eval_config["agent_to_test"], eval_config, train_config)

    steps_to_test = np.arange(eval_config["step_to_start"], eval_config["step_to_end"]+1, eval_config["step_interval"])
    all_returns = {}
    for step_nb in steps_to_test:
        print("Loading checkpoint…")
        checkpoint_path = op.join("evaluate_results", f"agent_{eval_config["agent_to_test"]}", "checkpoints", f"step_{step_nb}.pth")
        load_checkpoint(agent, checkpoint_path, eval_config)

        print("Running evaluation…")
        returns = evaluate(agent, eval_config, train_config)
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
    csv_dir = op.join("evaluate_results", f"agent_{eval_config["algorithm"]}") 
    Path(csv_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(op.join(csv_dir, 'evaluation_returns.csv'), index=False)

if __name__ == "__main__":
    # Load the test config yaml from in/config/default.yaml
    with open(op.join("in", "config", "eval_config.yaml"), 'r') as f:
        eval_config = yaml.safe_load(f)
    # Load the parameters from the training
    train_config = load_training_config(eval_config["agent_to_test"])
    # Run main
    main(eval_config, train_config)
