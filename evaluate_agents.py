import torch
import numpy as np
from pathlib import Path
import os.path as op
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

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

AGENT_TO_TEST = "logic_paper" # "logic_paper" or "ppo_paper" or "ppo_custom"
STEP_NB = 800001
CHECKPOINT = op.join("evaluate_results", f"agent_{AGENT_TO_TEST}", "checkpoints", f"step_{STEP_NB}.pth")
ENV_NAME = "getout"
DEVICE = "cuda:0"   # "cpu" or "cuda:0"
N_TEST = 50
TEST_SEED = 9999        # ensures reproducibility
ENV_KWARGS = {}          # fill if needed (same as during training)

# Hyperparameters MUST match the ones used to TRAIN the agent
RULES = "getout_bs_top10"
LR_ACTOR = 0.001
LR_CRITIC = 0.0003
GAMMA = 0.99
EPOCHS = 20
EPS_CLIP = 0.2
OPTIMIZER = torch.optim.Adam

# ---------------------------------------------------------
# Helper: build the agent the same way as in training
# ---------------------------------------------------------

def build_agent(env, agent_to_test):
    if agent_to_test == "logic_paper" or agent_to_test == "logic_rf10_paper":
        agent = LogicPPO(
            env=env,
            rules=RULES,
            lr_actor=LR_ACTOR,
            lr_critic=LR_CRITIC,
            optimizer=OPTIMIZER,
            gamma=GAMMA,
            epochs=EPOCHS,
            eps_clip=EPS_CLIP,
            device=DEVICE
        )
    elif agent_to_test == "ppo_paper":
        agent = NeuralPPO(
            env=env,
            lr_actor=LR_ACTOR,
            lr_critic=LR_CRITIC,
            optimizer=OPTIMIZER,
            gamma=GAMMA,
            epochs=EPOCHS,
            eps_clip=EPS_CLIP,
            device=DEVICE
        )
    return agent

# Helper to make the envs 
def make_env(seed: int, render: bool = False, plusplus: bool = False, noise: bool = False):
    """
    Factory to create NudgeEnv instances for vectorized environments.
    """

    def _init():
        env = env_module.NudgeEnv(
            mode="ppo",
            plusplus=plusplus,
            noise=noise,
            seed=seed)
        env.reset(seed=seed)
        return env

    return _init()


# ---------------------------------------------------------
# Load agent weights
# ---------------------------------------------------------

def load_checkpoint(agent, checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path} …")
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    # import ipdb; ipdb.set_trace()
    agent.policy.load_state_dict(state_dict)
    agent.policy_old.load_state_dict(state_dict)
    agent.policy.eval()
    agent.policy_old.eval()


# ---------------------------------------------------------
# Evaluate the agent
# ---------------------------------------------------------

def evaluate(agent):
    """
    Evaluate the agent at several checkpoint over N_TEST deterministic environments.
    """
    returns = []

    # Fix environment generation for reproducibility
    rng = np.random.default_rng(TEST_SEED)

    for i in tqdm(range(N_TEST)):
        # Create environment with deterministic seed
        env_seed = int(rng.integers(0, 100000))
        env = make_env(seed=env_seed, render=False)

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

    print(f"\nEvaluation complete over {N_TEST} fixed test envs.")
    print(f"Median return:  {median_return:.3f}")
    print(f"Average return: {avg_return:.3f}")
    print(f"Std return:     {std_return:.3f}")

    return returns


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main(agent_to_test, steps_to_test):
    print("Loading environment template…")
    base_env = NudgeBaseEnv.from_name(ENV_NAME, mode="logic", **ENV_KWARGS)

    print("Building agent…")
    agent = build_agent(base_env, agent_to_test)

    all_returns = {}
    for step_nb in steps_to_test:
        print("Loading checkpoint…")
        checkpoint_path = op.join("evaluate_results", f"agent_{agent_to_test}", "checkpoints", f"step_{step_nb}.pth")
        load_checkpoint(agent, checkpoint_path)

        print("Running evaluation…")
        returns = evaluate(agent)
        all_returns[step_nb] = returns
        # Plot returns distribution
        plt.figure(figsize=(8, 5))
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
    agent_to_test = "logic_rf10_paper" # "logic_paper" or "ppo_paper"
    steps_to_test = np.arange(1, 500002, 100000)
    main(agent_to_test, steps_to_test)
