import csv
import os
import sys
import time
from datetime import datetime
from inspect import signature
from pathlib import Path
from typing import Callable

import numpy as np
import yaml
from rtpt import RTPT
import torch
from torch.optim import Optimizer, Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nudge.agents.logic_agent import LogicPPO
from nudge.agents.neural_agent import NeuralPPO
from nudge.env import NudgeBaseEnv
from nudge.utils import make_deterministic, save_hyperparams
from nudge.utils import exp_decay
from nudge.utils import print_program
from argparse import ArgumentParser

### GC ### 
from env_src.getout.getout.goal_conduciveness import GoalConduciveness
# debug support # 
from env_src.getout.getout.state_debug_visual import log_entity_positions



OUT_PATH = Path("out/")
IN_PATH = Path("in/")


def main(algorithm: str,
         environment: str,
         env_kwargs: dict = None,
         rules: str = "default",
         seed: int = 0,
         device: str = "cpu",
         total_steps: int = 800000,
         max_ep_len: int = 500,
         update_steps: int = None,
         epochs: int = 20,
         eps_clip: float = 0.2,
         gamma: float = 0.99,
         optimizer: Optimizer = Adam,
         lr_actor: float = 0.001,
         lr_critic: float = 0.0003,
         epsilon_fn: Callable = exp_decay,
         recover: bool = False,
         save_steps: int = 250000,
         stats_steps: int = 2000,
         gc_gamma: float = 1.0,
         gc_normalize: bool = True,
         gc_update: str = "with_agent",
         ):
    """

    Args:
        algorithm: Either 'ppo' for Proximal Policy Optimization or 'logic'
            for First-Order Logic model
        environment: The name of the environment to use (prepared inside in/envs)
        env_kwargs: Optional settings for the environment
        rules: The name of the logic rule set to use
        seed: Random seed for reproduction
        device: For example 'cpu' or 'cuda:0'
        total_steps: Total number of time steps to train the agent
        max_ep_len: Maximum number of time steps per episode
        update_steps: Number of time steps between agent updates. Caution: if too
            high, causes OutOfMemory errors when running with CUDA.
        epochs: Number of epochs (k) per agent update
        eps_clip: Clipping factor epsilon for PPO
        gamma: Discount factor
        optimizer: The optimizer to use for agent updates
        lr_actor: Learning rate of the actor (policy)
        lr_critic: Learning rate of the critic (value fn)
        epsilon_fn: Function mapping episode number to epsilon (greedy) for
            exploration
        recover: If true, tries to reload an existing run that was interrupted
            before completion.
        save_steps: Number of steps between each checkpoint save
        stats_steps: Number of steps between each statistics summary timestamp
        gc_gamma: Discount factor for Goal Conduciveness potential shaping
        gc_normalize: Whether to normalize Goal Conduciveness score by subgoal count
        gc_update: When to append new subgoals ('with_agent' or 'episodic')
    """

    make_deterministic(seed)
    
    # device = torch.device("mps") if (torch.backends.mps.is_available() and device == "cpu") else "cpu"
    # device = torch.device("cuda") if torch.cuda.is_available() else device
    print(device)

    assert algorithm in ['ppo', 'logic']

    if env_kwargs is None:
        env_kwargs = dict()

    if update_steps is None:
        if algorithm == 'ppo':
            update_steps = max_ep_len * 4
        else:
            update_steps = 100

    env = NudgeBaseEnv.from_name(environment, mode=algorithm, **env_kwargs)

    now = datetime.now()
    dt = now.strftime('%y-%m-%d-%H-%M')
    gc_data = f"steps_{total_steps}_gamma{gc_gamma}_{'norm' if gc_normalize else 'denorm'}_{gc_update}"
    experiment_dir = OUT_PATH / "runs" / environment / f"{algorithm}_gc" / f"{dt}_{gc_data}"
    checkpoint_dir = experiment_dir / "checkpoints"
    image_dir = experiment_dir / "images"
    log_dir = experiment_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    save_hyperparams(signature=signature(main),
                     local_scope=locals(),
                     save_path=experiment_dir / "config.yaml",
                     print_summary=True)

    # initialize agent
    if algorithm == "ppo":
        agent = NeuralPPO(env, lr_actor, lr_critic, optimizer,
                          gamma, epochs, eps_clip, device)
    else:  # logic
        agent = LogicPPO(env, rules, lr_actor, lr_critic, optimizer,
                         gamma, epochs, eps_clip, device)
        print_program(agent)
        # print('Candidate Clauses:')
        # for clause in agent.policy.actor.clauses:
        #     print(clause)
        # print()

    i_episode = 0
    weights_list = []

    if recover:
        if algorithm == 'ppo':
            step_list, reward_list = agent.load(checkpoint_dir)
        else:  # logic
            step_list, reward_list, weights_list = agent.load(checkpoint_dir)
        time_step = max(step_list)[0]
    else:
        step_list = []
        reward_list = []
        time_step = 0

    # track total training time
    start_time = time.time()
    print("Started training at ", now.strftime("%H:%M"))

    # printing and logging variables
    running_ret = 0  # running return
    n_episodes = 0

    rtpt = RTPT(name_initials='HS', experiment_name='LogicRL',
                max_iterations=total_steps)

    # Start the RTPT tracking
    writer = SummaryWriter(str(log_dir))
    rtpt.start()
    
    """ Implementation of Goal Conduciveness """
    gc = GoalConduciveness(gamma=gc_gamma, normalize=gc_normalize, update=gc_update)
    """"""
    
    """ TEST Loading Trained GC Info """
    # gc_path = experiment_dir / "goal_conduciveness.yaml"
    # gc_path = IN_PATH / "config" / "GC_dummy.yaml"
    # if gc_path.exists():
    #     with open(gc_path, "r") as f:
    #         gc_payload = yaml.safe_load(f)
    #     gc.load_GC(gc_payload)
    # this could be included as experiment, i.e. training with pre-loaded goal-conduciveness appraisal
    """"""

    visual_state_debug = False  # set True to enable visual state debugging
    
    pbar = tqdm(total=total_steps - time_step, file=sys.stdout)
    while time_step < total_steps:
        state, state_variables = env.reset()
        ret = 0  # return
        n_episodes += 1
        epsilon = epsilon_fn(i_episode)
        
        """ Goal Conduciveness """
        # at start of each episode, reset GC progress for all subgoals, and activate first subgoal
        r_gc = 0.0
        r_gc_prev = 0.0
        gc.reset_GC_progress(state_variables)

        # if episodic update, introduce new subgoals here
        if gc.update == "episodic": 
            gc.append_queue()
        """"""
            
        # Play episode
        for t in range(max_ep_len):
            action = agent.select_action(state, epsilon=epsilon)

            state, state_variables, reward, done = env.step(action)
            base_reward = reward
            
            """ Goal Conduciveness """
            # if reward obtained, we need to check whether a new reward source has been found (ie a new subgoal)
            # ...or whether a current active subgoal has been completed
            if reward > 0:             
                # ugly but functional way to check for player collisions (ie determine source of reward)
                reward_sources = env.env.level.entities[0].collisions  # (key=0, door=1, enemy=2)
                if reward_sources[0] == True:
                    reward_source = 'key'
                elif reward_sources[1] == True: 
                    reward_source = 'door'
                else: 
                    raise ValueError("Unkown source")
                # try to add subgoal to subgoal queue (won't add if subgoal already in dict)
                if not gc.add_subgoal_to_queue(obj_type=reward_source): 
                    gc.complete_current_subgoal(reward_source, state_variables) 
                    # if not, the subgoal may have been active, and if so can be completed
            
            # compute active goal progress, then compute total GC score
            gc.compute_active_progress(state_variables)
            r_gc = gc.compute_GC_score()
                        
            potential_diff = gc.gamma * (r_gc - r_gc_prev)
            r_gc_prev = r_gc

            reward += potential_diff  # for higher resolution we may want to store both pre and post reward term buffer values 
            
            gc.display_GC() # debug purposes
            if visual_state_debug:
                log_entity_positions(
                    env,
                    action,
                    base_reward,
                    potential_diff=potential_diff,
                    r_gc=r_gc,
                    n_episodes=n_episodes,
                    step=getattr(env.env, "step_counter", time_step),
                )
            """"""


            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
            agent.buffer.r_gc.append(r_gc)  # added gc to buffer (maybe pointless?)


            time_step += 1
            pbar.update(1)
            rtpt.step()
            ret += reward

            if time_step % update_steps == 0: # backprop every #update_steps (def 100)
                agent.update()

                # if set to update when agent does, introduce new subgoals here
                if gc.update == "with_agent": 
                    gc.append_queue()

            # printing average reward
            if time_step % stats_steps == 0:
                # print average reward till last episode
                avg_return = running_ret / n_episodes
                avg_return = round(avg_return, 2)

                print(f"\nEpisode: {i_episode} \t\t Timestep: {time_step} \t\t Average Reward: {avg_return}")
                running_ret = 0
                n_episodes = 1

                step_list.append([time_step])
                reward_list.append([avg_return])
                writer.add_scalar('AvgReturn', avg_return, time_step)
                if algorithm == 'logic':
                    weights_list.append([(agent.get_weights().tolist())])

            # save model weights
            if time_step % save_steps == 1:
                checkpoint_path = checkpoint_dir / f"step_{time_step}.pth"
                if algorithm == 'logic':
                    agent.save(checkpoint_path, checkpoint_dir, step_list, reward_list, weights_list)
                else:
                    agent.save(checkpoint_path, checkpoint_dir, step_list, reward_list)
                print("\nSaved model at:", checkpoint_path)

            if done:
                break

        running_ret += ret
        i_episode += 1
        writer.add_scalar('Return', ret, i_episode)
        writer.add_scalar('ReturnPerStep', ret, time_step)
        writer.add_scalar('Epsilon', epsilon, i_episode)
        writer.add_scalar('EpisodeLength', t + 1, i_episode)
        writer.add_scalar('GoalConduciveness/score', r_gc, time_step)

    env.close()
    pbar.close()

    with open(experiment_dir / 'data.csv', 'w', newline='') as f:
        dataset = csv.writer(f)
        header = ('steps', 'reward')
        dataset.writerow(header)
        data = np.hstack((step_list, reward_list))
        for row in data:
            dataset.writerow(row)

    if algorithm == 'logic':
        with open(experiment_dir / 'weights.csv', 'w', newline='') as f:
            dataset = csv.writer(f)
            for row in weights_list:
                dataset.writerow(row)

    # Persist Goal Conduciveness state for reuse with a trained model
    gc_payload = gc.return_GC_info()
    with open(experiment_dir / "goal_conduciveness.yaml", "w") as f:
        yaml.safe_dump(gc_payload, f)


    end_time = time.time()
    print("Finished training at", datetime.now().strftime("%H:%M"))
    print(f"Total training time: {(end_time - start_time) / 60 :.0f} min")




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("-g", "--game", type=str, default=None)
    parser.add_argument("-d", "--device", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.config is None:
        config_path = IN_PATH / "config" / "logic_with_Goal_Conduciveness.yaml"
    else:
        config_path = Path(args.config)

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
        
    # Map string names to actual functions
    ## Epsilon function
    epsilon_fn_map = {
    "exp_decay": exp_decay,
    ### Add other functions here as needed
    }
    if "epsilon_fn" in config and config["epsilon_fn"] in epsilon_fn_map:
        config["epsilon_fn"] = epsilon_fn_map[config["epsilon_fn"]]
    ## Optimizer
    optimizer_map = {
        "Adam": Adam,
        ### Add other optimizers here as needed
    }
    if "optimizer" in config and config["optimizer"] in optimizer_map:
        config["optimizer"] = optimizer_map[config["optimizer"]]
    
    if args.game is not None:
        config["environment"] = args.game
    if args.device is not None:
        config["device"] = args.device

    main(**config)