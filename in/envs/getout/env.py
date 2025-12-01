from enum import Enum
from typing import Sequence
import sys
import os.path as op

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import torch

from nudge.utils import simulate_prob
from nudge.env import NudgeBaseEnv

# Make env_src importable
sys.path.append(op.abspath(op.join(op.dirname(__file__), "..", "..", "..", "env_src")))
from getout.getout.paramLevelGenerator import ParameterizedLevelGenerator


class NudgeEnv(NudgeBaseEnv, gym.Env):
    """
    Gym-compatible wrapper around the Getout environment.

    - Uses Getout as the underlying game.
    - Returns a logic-state observation (n_objects x n_features).
    - Can be used directly with Stable-Baselines3.
    """

    name = "getout"
    pred2action = {
        "stay": 0,
        "idle": 0,
        "left": 1,
        "right": 2,
        "jump": 3,
    }
    pred_names: Sequence

    metadata = {"render_modes": []}

    def __init__(self, mode: str, plusplus: bool = False, noise: bool = False, render: bool = False, seed=np.random.randint(1, 10000)):
        NudgeBaseEnv.__init__(self, mode)
        gym.Env.__init__(self)
        self.plusplus = plusplus
        self.noise = noise
        self.render_enabled = render

        # Register underlying game env only once
        try:
            register(id="getout", entry_point="env_src.getout.getout.getout:Getout")
        except gym.error.Error:
            pass  # already registered
        # Create underlying game environment
        self.env = gym.make("getout", render=render).unwrapped
        # Generate initial level
        self._generate_new_level(seed=seed)
        if self.render_enabled:
            self.env.render()
        # Logic-state dimensions
        self.n_features = 6
        self.n_objects = 8 if self.plusplus else 4

        # Define observation space
        self.observation_space = self.env.observation_space 
        # Action space 
        self.action_space = self.env.action_space

    # Helper function: generate new level
    def _generate_new_level(self, seed=np.random.randint(1, 10000)):
        level_generator = ParameterizedLevelGenerator(
            enemy=False, enemies=False, key_door=False
        )
        level_generator.generate(self.env, seed=seed)
        # level_generator.generate(self.env, seed=1)

    # Reset function
    def reset(self, seed=None, options=None):
        """
        Resets the environment and returns the logic-state observation.
        """
        gym.Env.reset(self, seed=seed)
        # Reset internal game env
        _, _ = self.env.reset(seed=seed)
        # Regenerate a new random level
        self._generate_new_level(seed=seed)
        if self.render_enabled:
            self.env.render()
        # Use env.get_obs() to get the structured obs dict
        # logic_state = self.extract_logic_state(self.env.get_obs())
        return self.env.get_obs(), {}

    # Step function
    def step(self, action, is_mapped: bool = False):
        """
        Step the environment.

        Returns:
            logic_state, reward, terminated, truncated, info
        """
        # If you ever want to use pred2action mapping, do it here when is_mapped=False.
        obs, reward, terminated, truncated, info = self.env.step(action)
        # logic_state = self.extract_logic_state(obs)
        return obs, reward, terminated, truncated, info

    # State extractor 
    def extract_state(self, observation):
        """
        Get the full internal representation (level + entities) from the base env.
        """
        repr = self.env.level.get_representation()
        repr["reward"] = observation["reward"]
        repr["score"] = observation["score"]

        for entity in repr["entities"]:
            # replace all enums (e.g. EntityID) with int values
            for i, v in enumerate(entity):
                if isinstance(v, Enum):
                    entity[i] = v.value
                if isinstance(v, bool):
                    entity[i] = int(v)

        return repr

    # Logic-state extractor 
    def extract_logic_state(self, observation):
        """
        Extracts the logic state from the raw environment observation.

        Args:
            observation: The raw observation dict from the environment.

        Returns:
            np.ndarray of shape (n_objects, n_features)
        """
        n_features = self.n_features
        n_objects = self.n_objects
        logic_state = np.zeros((n_objects, n_features), dtype=float)

        for key, value in observation.items():
            if key == "player":
                logic_state[0][0] = 1
                logic_state[0][-2:] = value
            elif key == "key":
                logic_state[1][1] = 1
                logic_state[1][-2:] = value
            elif key == "door":
                logic_state[2][2] = 1
                logic_state[2][-2:] = value
            elif key in ["enemy", "ground_enemy"]:
                logic_state[3][3] = 1
                logic_state[3][-2:] = value
            elif key == "ground_enemy2" and n_objects > 4:
                logic_state[4][3] = 1
                logic_state[4][-2:] = value
            elif key == "ground_enemy3" and n_objects > 5:
                logic_state[5][3] = 1
                logic_state[5][-2:] = value
            elif key == "buzzsaw1" and n_objects > 6:
                logic_state[6][3] = 1
                logic_state[6][-2:] = value
            elif key == "buzzsaw2" and n_objects > 7:
                logic_state[7][3] = 1
                logic_state[7][-2:] = value

        if self.noise:
            key_picked = sum(logic_state[:, 1]) == 0
            logic_state = simulate_prob(logic_state, n_objects, key_picked)

        return logic_state

    def extract_neural_state(self, observation):
        """
        Optional: build a neural-state tensor using your existing utilities.
        """
        model_input = sample_to_model_input((self.extract_state(observation), []))
        model_input = collate([model_input], to_cuda=False, double_to_float=True)
        neural_state = model_input["state"]
        neural_state = torch.cat([neural_state["base"], neural_state["entities"]], dim=1)
        return neural_state

    def close(self):
        self.env.close()

## Helper functions
def for_each_tensor(o, fn):
    if isinstance(o, torch.Tensor):
        return fn(o)
    if isinstance(o, list):
        for i, e in enumerate(o):
            o[i] = for_each_tensor(e, fn)
        return o
    if isinstance(o, dict):
        for k, v in o.items():
            o[k] = for_each_tensor(v, fn)
        return o
    raise ValueError("unexpected object type")


def collate(samples, to_cuda=True, double_to_float=True):
    samples = torch.utils.data._utils.collate.default_collate(samples)
    if double_to_float:
        samples = for_each_tensor(
            samples,
            lambda tensor: tensor.float() if tensor.dtype == torch.float64 else tensor,
        )
    if to_cuda:
        samples = for_each_tensor(samples, lambda tensor: tensor.cuda())
    return samples


def sample_to_model_input(sample, no_dict=False, include_score=False):
    """
    :param sample:  tuple: (representation (use extract_state), explicit_action (unused here))
    :return: {state: {base:[...], entities:[...]}, action:0..9}
    """
    state = sample[0]
    action = sample[1]

    tr_entity, swap_coins = fixed_size_entity_representation(state)
    tr_entities, swap_coins = replace_bools(tr_entity, swap_coins)

    if no_dict:
        # ignores the action and returns a single [60] array
        return [
            0,
            0,
            state["level"]["reward_key"],
            state["level"]["reward_powerup"],
            state["level"]["reward_enemy"],
            state["score"] if include_score else 0,
            *tr_entities,
        ]

    tr_state = {
        "base": np.array(
            [
                0,
                0,
                state["level"]["reward_key"],
                state["level"]["reward_powerup"],
                state["level"]["reward_enemy"],
                state["score"] if include_score else 0,
            ]
        ),
        "entities": np.array(tr_entities),
    }

    return {
        "state": tr_state,
        "action": action,
    }


ENTITY_ENCODING_LENGTH = 9
COIN0_IDX = 4 * ENTITY_ENCODING_LENGTH
COIN1_IDX = 5 * ENTITY_ENCODING_LENGTH
IDX_X = 1


def fixed_size_entity_representation(state, swap_coins=None):
    """
    Compresses the list of entities into an fixed size array (MAX_ENTITIES(6)*ENTITY_ENCODING(9))
    Entity order: player, flag, powerup, enemy, coin0, coin1.
    Entity encoding: [x,y, vx,vy, E0..E3]
    """

    entities = state["entities"]
    MAX_ENTITIES = 6  # player, flag, powerup, enemy, 2*coin
    ENTITY_ENCODING = 9
    tr_entities = [0] * ENTITY_ENCODING * MAX_ENTITIES

    coin_count = 0
    # IDs: PLAYER = 1, FLAG = 2, COIN = 3, POWERUP = 4, GROUND_ENEMY = 5
    # Position encoding: player, flag, powerup, enemy, coin0, coin1
    for entity in entities:
        id_ = entity[0]
        if id_ == 3:
            id_ = 5 + coin_count
            coin_count += 1
        elif id_ > 3:
            id_ -= 1
        id_ -= 1

        start_pos = id_ * ENTITY_ENCODING
        tr_entities[start_pos : start_pos + ENTITY_ENCODING] = entity

    if swap_coins is None:
        swap_coins = (
            coin_count == 2
            and tr_entities[COIN1_IDX + IDX_X] < tr_entities[COIN0_IDX + IDX_X]
        )

    if swap_coins:
        # swap coins so coin0 is left-most
        temp_coin = tr_entities[COIN0_IDX : COIN0_IDX + ENTITY_ENCODING_LENGTH]
        tr_entities[COIN0_IDX : COIN0_IDX + ENTITY_ENCODING_LENGTH] = tr_entities[
            COIN1_IDX : COIN1_IDX + ENTITY_ENCODING_LENGTH
        ]
        tr_entities[COIN1_IDX : COIN1_IDX + ENTITY_ENCODING_LENGTH] = temp_coin

    return tr_entities, swap_coins


def replace_bools(tr_entities, swap_coins):
    swap_coins = int(swap_coins)
    return tr_entities, swap_coins