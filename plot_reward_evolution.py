import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as op

# Plot the evolution of rewards over training steps
# Columns contain the reward for the agent trained for that amount of timesteps
# The first line contains the train timestep #
def plot_reward_evolution(data_path: str, output_path: str):
    # Load data
    data = pd.read_csv(data_path, index_col=0)
    # Get per column statistics
    means = data.mean()
    stds = data.std()
    steps = means.index.astype(int)
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, means, label='Average Return', color='blue')
    plt.fill_between(steps, means - stds, means + stds, color='blue', alpha=0.2, label='Std Dev')
    plt.title("Evolution of custom PPO's average evaluation reward over training steps (N_eval_envs = 50)")
    plt.xlabel('Training Steps')
    plt.ylabel('Average Return')
    plt.legend()
    plt.grid()
    plt.savefig(output_path)


if __name__ == "__main__":
    data_path = op.join("evaluate_results", "agent_ppo_custom", "evaluation_returns.csv")
    output_path = op.join("evaluate_results", "agent_ppo_custom", "plots", "return_stats.png")
    plot_reward_evolution(data_path, output_path)