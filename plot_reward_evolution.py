import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as op
import ast

AGENT_NAMES_TO_PLOT = [
    # "agent_logic_rf1",
    # "agent_logic_rf10",
    # "agent_logic_test",
    # "agent_ppo_1500k",
    # "agent_ppo_custom",
    # "agent_ppo_final_2M",
    "agent_ppo_final_5M",
    # "agent_ppo_new",
    # "agent_ppo_new_3500k",
    # "agent_ppo_old_2M",
    # "agent_ppo_paper"
    "agent_ppo_gc_gamma6",
    "agent_ppo_gc_gamma12",
    "agent_ppo_gc_gamma18",
]

colors = {
    # "agent_logic_rf1": "blue",
    # "agent_logic_rf10": "cyan",
    # "agent_ppo_test": "orange",
    # "agent_ppo_1500k": "green",
    # "agent_ppo_custom": "red",
    # "agent_ppo_final_2M": "purple",
    "agent_ppo_final_5M": "brown",
    # "agent_ppo_new": "pink",
    # "agent_ppo_new_3500k": "gray",
    # "agent_ppo_old_2M": "olive",
    # "agent_ppo_paper": "magenta",
    "agent_ppo_gc_gamma6": "green",
    "agent_ppo_gc_gamma12": "orange",
    "agent_ppo_gc_gamma18": "pink",
}

labels = {
    # "agent_logic_rf1": "Logic RF1",
    # "agent_logic_rf10": "Logic RF10",
    # "agent_ppo_test": "orange",
    # "agent_ppo_1500k": "PPO 1.5M",
    # "agent_ppo_custom": "Handcrafted PPO with compact 6D observation",
    # "agent_ppo_final_2M": "Run 1",
    "agent_ppo_final_5M": "Non-GC-PPO",
    # "agent_ppo_new": "PPO New",
    # "agent_ppo_new_3500k": "PPO New 3.5M",
    # "agent_ppo_old_2M": "PPO Old 2M",
    # "agent_ppo_paper": "PPO from Delfosse et al. with 60D observation",
    "agent_ppo_gc_gamma6": "GC-PPO with γ=6",
    "agent_ppo_gc_gamma12": "GC-PPO with γ=12",
    "agent_ppo_gc_gamma18": "GC-PPO with γ=18",
}

plot_training = False

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
        if agent_to_plot in ["agent_ppo_old_2M", "agent_ppo_final_5M", "agent_logic_rf10", "agent_ppo_gc_gamma6", "agent_ppo_gc_gamma12", "agent_ppo_gc_gamma18"]:
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
        if plot_training:
            data_training = pd.read_csv(op.join("evaluate_results", agent_to_plot, "data.csv"))
            train_steps = data_training["steps"][::20]
            train_rewards = data_training["reward"][::20]
            # plot training rewards as a light line
            plt.plot(train_steps, train_rewards, label=f"{labels[agent_to_plot]} (training)", color=colors[agent_to_plot], alpha=0.3)
        # Plot
        plt.plot(steps, means, label=labels[agent_to_plot], color=colors[agent_to_plot])
        # plt.fill_between(steps, means - stds, means + stds, color=colors[agent_to_plot], alpha=0.2)
    plt.title("Average evaluation reward for different algorithms across training steps (N_eval_envs = 50)")
    plt.xlabel('Training Steps')
    plt.ylabel('Average Return')
    plt.ticklabel_format(style='plain', axis='x')
    plt.xticks(np.arange(0, 5000001, 200000),
            [f"{x//1000}k" for x in np.arange(0, 5000001, 200000)],
            rotation=45, ha='right')
    plt.xlim(0, 5000000)
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
        if agent_to_plot in ["agent_ppo_old_2M", "agent_ppo_final_5M", "agent_logic_rf10", "agent_ppo_gc_gamma6", "agent_ppo_gc_gamma12", "agent_ppo_gc_gamma18"]:
            completions = data["completions"]
            completions = completions.apply(ast.literal_eval)
            completion_rates = completions.apply(np.mean) * 100  # percentage of environments completed
            steps = completion_rates.index.astype(int)*100000
        else:
            completion_rates = (data >= 0).mean() * 100  # percentage of environments completed
            steps = completion_rates.index.astype(int)
        # Plot
        plt.plot(steps, completion_rates, label=labels[agent_to_plot], color=colors[agent_to_plot])
    plt.title("Level completion rate for different algorithms across training steps (N_eval_envs = 50)")
    plt.xlabel('Training Steps')
    plt.ylabel('Completion Rate (%)')
    plt.ticklabel_format(style='plain', axis='x')
    plt.xticks(np.arange(0, 5000001, 200000),
            [f"{x//1000}k" for x in np.arange(0, 5000001, 200000)],
            rotation=45, ha='right')
    plt.xlim(0, 5000000)
    plt.legend()
    plt.grid()
    plt.savefig(output_path)


if __name__ == "__main__":
    output_path_rewards = op.join("evaluate_results", "reward_stats.png")
    output_path_completion = op.join("evaluate_results", "completion_stats.png")
    plot_reward_evolution(AGENT_NAMES_TO_PLOT, output_path_rewards)
    plot_completion_rate(AGENT_NAMES_TO_PLOT, output_path_completion)