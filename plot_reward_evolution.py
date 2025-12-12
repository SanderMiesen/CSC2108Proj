import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as op

AGENT_NAMES_TO_PLOT = [
    "agent_ppo_custom",
    "agent_logic_rf10_paper",
    "agent_logic_paper",
    "agent_ppo_paper"
]

colors = {
    "agent_ppo_custom": "orange",
    "agent_logic_rf10_paper": "green",
    "agent_logic_paper": "blue",
    "agent_ppo_paper": "red"
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
        # Get per column statistics
        means = data.mean()
        stds = data.std()
        steps = means.index.astype(int)
        # Plot
        plt.plot(steps, means, label=agent_to_plot, color=colors[agent_to_plot])
        # plt.fill_between(steps, means - stds, means + stds, color=colors[agent_to_plot], alpha=0.2)
    plt.title("Evolution of methods' average evaluation reward over training steps (N_eval_envs = 50)")
    plt.xlabel('Training Steps')
    plt.ylabel('Average Return')
    plt.ticklabel_format(style='plain', axis='x')
    plt.xticks(np.arange(0, 1800001, 100000),
            [f"{x//1000}k" for x in np.arange(0, 1800001, 100000)],
            rotation=45, ha='right')
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
        completion_rates = (data >= 0).mean() * 100  # percentage of environments completed
        steps = completion_rates.index.astype(int)
        # Plot
        plt.plot(steps, completion_rates, label=agent_to_plot, color=colors[agent_to_plot])
    plt.title("Evolution of methods' level completion rate over training steps (N_eval_envs = 50)")
    plt.xlabel('Training Steps')
    plt.ylabel('Completion Rate (%)')
    plt.ticklabel_format(style='plain', axis='x')
    plt.xticks(np.arange(0, 1800001, 100000),
            [f"{x//1000}k" for x in np.arange(0, 1800001, 100000)],
            rotation=45, ha='right')
    plt.legend()
    plt.grid()
    plt.savefig(output_path)


if __name__ == "__main__":
    output_path_rewards = op.join("evaluate_results", "reward_stats.png")
    output_path_completion = op.join("evaluate_results", "completion_stats.png")
    plot_reward_evolution(AGENT_NAMES_TO_PLOT, output_path_rewards)
    plot_completion_rate(AGENT_NAMES_TO_PLOT, output_path_completion)