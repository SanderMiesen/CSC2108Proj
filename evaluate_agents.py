import torch
import numpy as np
from pathlib import Path
import os.path as op

import nudge
# from nudge.utils import make_deterministic
from nudge.agents.logic_agent import LogicPPO

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

AGENT_TO_TEST = "logic" # "logic" or "ppo_paper" or "ppo_custom"
STEP_NB = 800001
CHECKPOINT = op.join("evaluate_results", f"agent_{AGENT_TO_TEST}", "checkpoints", f"step_{STEP_NB}.pth")
ENV_NAME = "getout"
RULES = "default"
DEVICE = "cpu"   # or "cuda:0"
N_TEST = 100
TEST_SEED = 9999        # ensures reproducibility
ENV_KWARGS = {}          # fill if needed (same as during training)

# PPO hyperparameters MUST match the ones used to TRAIN the agent
LR_ACTOR = 0.001
LR_CRITIC = 0.0003
GAMMA = 0.99
EPOCHS = 20
EPS_CLIP = 0.2
OPTIMIZER = torch.optim.Adam

# ---------------------------------------------------------
# Helper: build the agent the same way as in training
# ---------------------------------------------------------

def build_agent(env):
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
    return agent


# ---------------------------------------------------------
# Load agent weights
# ---------------------------------------------------------

def load_checkpoint(agent, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    agent.policy.load_state_dict(state_dict)
    agent.policy_old.load_state_dict(state_dict)
    agent.policy.eval()
    agent.policy_old.eval()


# ---------------------------------------------------------
# Evaluate the agent
# ---------------------------------------------------------

def evaluate(agent):
    """
    Evaluate one of the agents at one checkpoint over N_TEST deterministic environments.
    """
    returns = []

    # Fix environment generation for reproducibility
    rng = np.random.default_rng(TEST_SEED)

    for i in range(N_TEST):
        # Create environment with deterministic seed
        env_seed = int(rng.integers(0, 100000))
        env = nudge.env.NudgeBaseEnv.from_name(
            ENV_NAME, 
            mode="logic", 
            seed=env_seed,
            **ENV_KWARGS
        )

        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action_pred = agent.select_action(state, epsilon=0.0)
            state, reward, done = env.step(action_pred)
            total_reward += reward

        returns.append(total_reward)
        env.close()

    avg_return = np.mean(returns)
    std_return = np.std(returns)

    print(f"\nEvaluation complete over {N_TEST} fixed test envs.")
    print(f"Average return: {avg_return:.3f}")
    print(f"Std return:     {std_return:.3f}")

    return returns


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print("Loading environment template…")
    base_env = nudge.env.NudgeBaseEnv.from_name(ENV_NAME, mode="logic", **ENV_KWARGS)

    print("Building agent…")
    agent = build_agent(base_env)

    print("Loading checkpoint…")
    load_checkpoint(agent, CHECKPOINT)

    print("Running evaluation…")
    evaluate(agent)


if __name__ == "__main__":
    main()
