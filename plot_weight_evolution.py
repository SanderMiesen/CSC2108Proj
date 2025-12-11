import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import ast

weights_csv_path = op.join("evaluate_results", f"agent_logic_rf10_paper", "weights.csv")
plot_save_path = op.join("evaluate_results", f"agent_logic_rf10_paper", "weight_evolution.png")

# Load the CSV file containing evaluation returns
def load_evaluation_data(path):
    data = pd.read_csv(path)
    return data

def plot_weight_evolution(weights_csv_path: str, plot_save_path: str):
    weights_data = pd.read_csv(weights_csv_path, header=None)

    # Parse each cell safely
    def parse_cell(x):
        try:
            return ast.literal_eval(x)
        except Exception as e:
            print("Failed to parse row:", x)
            raise e

    parsed = weights_data[0].apply(parse_cell)
    weights_array = np.array(parsed.tolist())

    num_steps = weights_array.shape[0]
    num_weights = weights_array.shape[1] * weights_array.shape[2]  # flatten the 5×5

    steps = np.arange(1, num_steps + 1) * 2000

    plt.figure(figsize=(12, 8))
    flat_weights = weights_array.reshape(num_steps, -1)

    for i in range(num_weights):
        plt.plot(steps, flat_weights[:, i], label=f'W{i+1}')

    plt.xlabel('Training Steps')
    plt.ylabel('Weight Value')
    plt.title('Evolution of Weights Over Training Steps')
    plt.legend()
    plt.grid()
    plt.savefig(plot_save_path)
    plt.close()

def plot_initial_weights(weights_csv_path: str):
    weights_data = pd.read_csv(weights_csv_path, header=None)

    def parse_cell(x):
        return ast.literal_eval(x)

    # Parse first row
    parsed = weights_data.iloc[0].apply(parse_cell)

    # parsed[i] is a 5x5 matrix; expand them
    expanded_groups = []
    for cell in parsed.tolist():
        mat = np.array(cell)       # shape (5,5)
        expanded_groups.append(mat)

    # Stack into groups of 5 weights
    weights_array = np.vstack(expanded_groups)   # shape: (num_cells*5, 5)

    num_groups = weights_array.shape[0]
    x = np.arange(1, 51)

    plt.figure(figsize=(15, 10))

    for i in range(num_groups):
        plt.subplot(2, 3, i + 1)
        plt.bar(x, weights_array[i])
        plt.title(f'Initial Weights Group {i + 1}')
        plt.xlabel('Weight Index')
        plt.ylabel('Weight Value')
        plt.xticks(x)

    plt.tight_layout()
    initial_plot_path = op.join("evaluate_results", "agent_logic_rf10_paper", "initial_weights.png")
    plt.savefig(initial_plot_path)
    plt.close()

def plot_final_weights(weights_csv_path: str):
    weights_data = pd.read_csv(weights_csv_path, header=None)

    def parse_cell(x):
        return ast.literal_eval(x)

    # Parse first row
    parsed = weights_data.iloc[-1].apply(parse_cell)

    # parsed[i] is a 5x5 matrix; expand them
    expanded_groups = []
    for cell in parsed.tolist():
        mat = np.array(cell)       # shape (5,5)
        expanded_groups.append(mat)

    # Stack into groups of 5 weights
    weights_array = np.vstack(expanded_groups)   # shape: (num_cells*5, 5)

    num_groups = weights_array.shape[0]
    x = np.arange(1, 51)

    plt.figure(figsize=(15, 10))

    for i in range(num_groups):
        plt.subplot(2, 3, i + 1)
        print(weights_array[i])
        plt.bar(x, weights_array[i])
        plt.title(f'final Weights Group {i + 1}')
        plt.xlabel('Weight Index')
        plt.ylabel('Weight Value')
        plt.xticks(x)

    plt.tight_layout()
    initial_plot_path = op.join("evaluate_results", "agent_logic_rf10_paper", "final_weights.png")
    plt.savefig(initial_plot_path)
    plt.close()


if __name__ == "__main__":
    plot_weight_evolution(weights_csv_path, plot_save_path)
    plot_initial_weights(weights_csv_path)
    plot_final_weights(weights_csv_path)