import argparse
import os
from typing import Tuple

import torch
from torch.optim import Adam

from nudge.agents.logic_agent import LogicPPO
from nudge.env import NudgeBaseEnv
from nudge.utils import make_deterministic


DEFAULT_LOGIC_MODELS = {
    # beam_search_top1 for getout was trained with the rf1 rule set (5 clauses)
    "getout": ("results/models/getout/logic/beam_search_top1.pth", "getout_bs_rf1"),
    "threefish": ("results/models/threefish/logic/beam_search_top1.pth", "threefish_human_assisted"),
    "loot": ("results/models/loot/logic/beam_search_top1.pth", "loot_human_assisted"),
}


def load_agent(env_name: str, rules: str, model_path: str, device: str) -> Tuple[LogicPPO, NudgeBaseEnv]:
    env = NudgeBaseEnv.from_name(env_name, mode="logic")
    agent = LogicPPO(env, rules, lr_actor=1e-3, lr_critic=3e-4,
                     optimizer=Adam, gamma=0.99, epochs=20, eps_clip=0.2,
                     device=device)
    state_dict = torch.load(model_path, map_location=device)
    agent.policy.load_state_dict(state_dict)
    agent.policy_old.load_state_dict(state_dict)
    agent.policy.actor.eval()
    agent.policy.critic.eval()
    return agent, env


def eval_agent(agent: LogicPPO, env: NudgeBaseEnv, episodes: int, max_ep_len: int, epsilon: float) -> float:
    total = 0.0
    for _ in range(episodes):
        state = env.reset()
        ret = 0.0
        for _ in range(max_ep_len):
            action = agent.select_action(state, epsilon=epsilon)
            state, reward, done = env.step(action)
            ret += reward
            if done:
                break
        total += ret
    return total / episodes


def main():
    parser = argparse.ArgumentParser(description="Play/evaluate a pretrained logic agent.")
    parser.add_argument("--env", required=True, choices=list(DEFAULT_LOGIC_MODELS.keys()))
    parser.add_argument("--rules", help="Ruleset name used during training.")
    parser.add_argument("--model", help="Path to logic .pth state_dict.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--max-ep-len", type=int, default=500, help="Max timesteps per episode.")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Epsilon for eval (0 = greedy).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    make_deterministic(args.seed)

    default_model, default_rules = DEFAULT_LOGIC_MODELS[args.env]
    model_path = args.model or default_model
    rules = args.rules or default_rules
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    rules_dir = os.path.join("in", "envs", args.env, "logic", rules)
    if not os.path.isdir(rules_dir):
        raise FileNotFoundError(f"Ruleset directory not found: {rules_dir}")

    agent, env = load_agent(args.env, rules, model_path, args.device)
    avg_return = eval_agent(agent, env, args.episodes, args.max_ep_len, args.epsilon)
    env.close()
    print(f"Avg return over {args.episodes} eps: {avg_return:.2f}")


if __name__ == "__main__":
    main()
