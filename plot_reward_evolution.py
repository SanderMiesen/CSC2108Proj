import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as op
import ast

AGENT_NAMES_TO_PLOT = [
    "agent_ppo_custom",
    "agent_logic_rf10",
    "agent_logic_rf1",
    "agent_ppo_old",
    # "agent_ppo_new",
    # "agent_ppo_1500k",
    "agent_ppo_old_2M",
    # "agent_ppo_new_3500k",
    "agent_ppo_final_5M"
]

colors = {
    "agent_ppo_custom": "orange",
    "agent_logic_rf10": "green",
    "agent_logic_rf1": "blue",
    "agent_ppo_old": "red",
    # "agent_ppo_new": "purple",
    # "agent_ppo_1500k": "brown",
    "agent_ppo_old_modified_2M": "cyan",
    # "agent_ppo_new_3500k": "magenta",
    "agent_ppo_final_5M": "red"
}

# Plot the evolution of rewards over training steps
# Columns contain the reward for the agent trained for that amount of timesteps
# The first line contains the train timestep #
def plot_reward_evolution(agent_names_to_plot: str, output_path: str):
    plt.figure(figsize=(10, 6))
    for agent_to_plot in agent_names_to_plot:
        # Load data
        data_path = op.join("evaluate_results", agent_to_plot, "evaluation_returns.csv")
        data = pd.read_csv(data_path)
        # Get returns 
        if agent_to_plot in ["agent_ppo_old_2M", "agent_ppo_final_5M"]:
            returns = data["returns"]
            returns = returns.apply(ast.literal_eval)
            means = returns.apply(np.mean)
            stds = returns.apply(np.std)
            steps = means.index.astype(int)*100000
        else:
            returns = data
            means = returns.mean()
            stds = returns.std()
            steps = means.index.astype(int)
        # Plot
        plt.plot(steps, means, label=agent_to_plot[6:], color=colors[agent_to_plot])
        # plt.fill_between(steps, means - stds, means + stds, color=colors[agent_to_plot], alpha=0.2)
    plt.title("Evolution of methods' average evaluation reward over training steps (N_eval_envs = 50)")
    plt.xlabel('Training Steps')
    plt.ylabel('Average Return')
    plt.ticklabel_format(style='plain', axis='x')
    plt.xticks(np.arange(0, 3600001, 200000),
            [f"{x//1000}k" for x in np.arange(0, 3600001, 200000)],
            rotation=45, ha='right')
    plt.xlim(0, 3500000)
    plt.legend()
    plt.grid()
    plt.savefig(output_path)

# Plot the level completion rate for each agent
# A level is completed if the return is >= 0
def plot_completion_rate(agent_names_to_plot: str, output_path: str):
    plt.figure(figsize=(10, 6))
    for agent_to_plot in agent_names_to_plot:
        # Load data
        data_path = op.join("evaluate_results", agent_to_plot, "evaluation_returns.csv")
        data = pd.read_csv(data_path)
        # Compute completion rates
        if agent_to_plot in ["agent_ppo_old_2M", "agent_ppo_final_5M"]:
            completions = data["completions"]
            completions = completions.apply(ast.literal_eval)
            completion_rates = completions.apply(np.mean) * 100  # percentage of environments completed
            steps = completion_rates.index.astype(int)*100000
        else:
            completion_rates = (data >= 0).mean() * 100  # percentage of environments completed
            steps = completion_rates.index.astype(int)
        # Plot
        plt.plot(steps, completion_rates, label=agent_to_plot[6:], color=colors[agent_to_plot])
    plt.title("Evolution of methods' level completion rate over training steps (N_eval_envs = 50)")
    plt.xlabel('Training Steps')
    plt.ylabel('Completion Rate (%)')
    plt.ticklabel_format(style='plain', axis='x')
    plt.xticks(np.arange(0, 3600001, 200000),
            [f"{x//1000}k" for x in np.arange(0, 3600001, 200000)],
            rotation=45, ha='right')
    plt.xlim(0, 3500000)
    plt.legend()
    plt.grid()
    plt.savefig(output_path)


if __name__ == "__main__":
    output_path_rewards = op.join("evaluate_results", "reward_stats.png")
    output_path_completion = op.join("evaluate_results", "completion_stats.png")
    plot_reward_evolution(AGENT_NAMES_TO_PLOT, output_path_rewards)
    plot_completion_rate(AGENT_NAMES_TO_PLOT, output_path_completion)