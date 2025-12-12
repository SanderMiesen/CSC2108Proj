"""
Skeleton for loading a trained agent along with its Goal Conduciveness (GC) state.
Adapt paths and hyperparameters as needed; this is meant as a starting point.
"""

from pathlib import Path
import yaml
import torch
from torch.optim import Adam

from nudge.env import NudgeBaseEnv
from nudge.agents.neural_agent import NeuralPPO
from nudge.agents.logic_agent import LogicPPO
from nudge.utils import get_most_recent_checkpoint_step
from env_src.getout.getout.goal_conduciveness import GoalConduciveness


def load_trained_gc_run(run_dir: str | Path, device: str = "cpu"):
    """
    Load env, agent weights, and GC metadata from a completed training run.
    Assumes the run directory contains config.yaml, checkpoints/, and goal_conduciveness.yaml.
    """
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"
    gc_path = run_dir / "goal_conduciveness.yaml"
    checkpoint_dir = run_dir / "checkpoints"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    algorithm = cfg["algorithm"]
    environment = cfg["environment"]
    env_kwargs = cfg.get("env_kwargs", {}) or {}
    optimizer_cls = {"Adam": Adam}.get(cfg.get("optimizer"), Adam)

    env = NudgeBaseEnv.from_name(environment, mode=algorithm, **env_kwargs)

    # Instantiate agent the same way training did
    if algorithm == "ppo":
        agent = NeuralPPO(
            env,
            cfg.get("lr_actor", 0.001),
            cfg.get("lr_critic", 0.0003),
            optimizer_cls,
            cfg.get("gamma", 0.99),
            cfg.get("epochs", 20),
            cfg.get("eps_clip", 0.2),
            device,
        )
    else:
        agent = LogicPPO(
            env,
            cfg.get("rules", "default"),
            cfg.get("lr_actor", 0.001),
            cfg.get("lr_critic", 0.0003),
            optimizer_cls,
            cfg.get("gamma", 0.99),
            cfg.get("epochs", 20),
            cfg.get("eps_clip", 0.2),
            device,
        )

    # Load the most recent checkpoint weights
    latest_step = get_most_recent_checkpoint_step(checkpoint_dir)
    checkpoint_path = checkpoint_dir / f"step_{latest_step}.pth"
    state_dict = torch.load(checkpoint_path, map_location=device)
    agent.policy_old.load_state_dict(state_dict)
    agent.policy.load_state_dict(state_dict)

    # Load Goal Conduciveness metadata
    gc = GoalConduciveness(
        gamma=cfg.get("gc_gamma", 1.0),
        normalize=cfg.get("gc_normalize", True),
        update=cfg.get("gc_update", "with_agent"),
    )
    if gc_path.exists():
        with open(gc_path, "r") as f:
            gc_payload = yaml.safe_load(f)
        gc.load_GC(gc_payload)

    return env, agent, gc


def rollout_one_episode(env, agent, gc, epsilon: float = 0.0, max_ep_len: int = 500):
    """
    Minimal evaluation loop showing how to reuse the GC object alongside the agent.
    Returns the shaped return.
    """
    (state, state_variables) = env.reset()
    gc.reset_GC_progress(state_variables)
    r_gc_prev = 0.0
    ret = 0.0

    for _ in range(max_ep_len):
        action = agent.select_action(state, epsilon=epsilon)
        state, state_variables, reward, done = env.step(action)

        gc.compute_active_progress(state_variables)
        r_gc = gc.compute_GC_score()
        potential_diff = gc.gamma * (r_gc - r_gc_prev)
        r_gc_prev = r_gc

        ret += reward + potential_diff
        if done:
            break

    return ret


if __name__ == "__main__":
    # Example usage; swap in your run directory path
    env, agent, gc = load_trained_gc_run("out/runs/getout/logic_gc/example_run")
    episode_return = rollout_one_episode(env, agent, gc, epsilon=0.0, max_ep_len=500)
    print(f"Episode return (with GC shaping): {episode_return:.3f}")
