import pathlib
import gymnasium as gym
import numpy as np

from .actions import GetoutActions
from .camera import Camera
from .hud import HUD
from .level import Level
from .trackingCamera import TrackingCamera
from .resource_loader import ResourceLoader
from .player_v1 import Player
from gymnasium.spaces import Discrete, Box, Dict

from .entityEncoding import EntityID


class Getout(gym.Env):

    def __init__(self, render=True, resource_path=None, start_on_first_action=False, width=50, seed=None):
        # self.unwrapped = self 
        # 4 actions: jump, left, right, reset
        self.action_space = Discrete(4)
        # Observation space: player x,y ; enemy x,y ; key x,y ; door x,y 
        self.observation_space = Dict(
            {
                "player": Box(low=np.array([0,0], dtype=float), 
                              high=np.array([width,16], dtype=float)),
                "enemy": Box(low=np.array([0,0], dtype=float), 
                             high=np.array([width,16], dtype=float)),
                "key": Box(low=np.array([0,0], dtype=float), 
                           high=np.array([width,16], dtype=float)),
                "door": Box(low=np.array([0,0], dtype=float), 
                            high=np.array([width,16], dtype=float)),
            }
        )

        self.zoom = 42 - width//2

        # if resource_path is None:
        #     resource_path = pathlib.Path(__file__).joinpath('../../assets/kenney/')
        base = pathlib.Path(__file__).resolve().parent.parent  # env_src/getout
        resource_path = base / 'assets' / 'kenney'
        self.resource_loader = ResourceLoader(path=resource_path, sprite_size=self.zoom,
                                              no_loading=not render) if render else None

        """ change here to choose  mode """

        self.score = 0.0
        self.level = Level(width, 16)
        self.player = Player(self.level, 2, 2, self.resource_loader)
        self.level.entities.append(self.player)
        self.width = width

        # self.camera = TrackingCamera(900, 600, self.player, zoom=self.zoom) if render else None
        self.camera = Camera(900, 600, x=-10, y=-50, zoom=self.zoom) if render else None
        self.show_step_counter = False
        self.hud = HUD(self, self.resource_loader)
        # start stepping the game only after the first action.
        # this is useful when recording human run-throughs to avoid
        self.start_on_first_action = start_on_first_action
        self.has_started = not start_on_first_action

        self.step_counter = 0

    def clear(self):
        raise NotImplementedError()

    def step(self, action):
        """
        Step the environment by one timestep.

        Args:
            action: The action to take.

        Returns:
            observation: The next observation.
            reward: The reward obtained.
            terminated: Whether the episode has ended.
            truncated: Whether the episode was truncated.
            info: Additional information.
        """
        # Init terminated and truncated flags
        terminated = False 
        truncated = False
        # Start stepping the game only after the first action.
        if self.start_on_first_action and not self.has_started:
            if isinstance(action, int):
                la = 0
            else:
                la = len(action)
            if la == 0 or (la == 1 and action[0] == GetoutActions.NOOP.value):
                return None, None, None, None, self.get_info()
            else:
                self.has_started = True
        # Increment step counter
        self.step_counter += 1
        # Apply action and step the level
        self.player.set_action(action)
        self.level.step()
        # Render the environment
        self.render()
        # If we want to add episode truncation on score <= 0, uncomment this
        """
        if self.score + reward <= 0 and not self.level.terminated:
            # terminate if the score drops below zero
            self.level.terminate(True)
            truncated = True
        """
        # Check if the level has terminated after the step
        if self.level.terminated:
            terminated = True
        # Get reward and update score
        reward = self.level.get_reward()
        self.score += reward

        return self.get_obs(), reward, terminated, truncated, self.get_info()

    def render(self):
        if self.camera is not None:
            self.camera.start_render()
            self.level.render(self.camera, self.step_counter)
            self.camera.end_render()

            self.hud.render(self.camera, step=self.step_counter if self.show_step_counter else None)

    def get_score(self):
        return self.score

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns that state.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration, unused here.

        Returns:
            tuple: (observation, info) for the initial state
        """
        super().reset(seed=seed)
        # import ipdb; ipdb.set_trace()
        # Reset everything without creating a new instance of Getout
        self.score = 0.0
        self.level = Level(self.width, 16)
        self.player = Player(self.level, 2, 2, self.resource_loader)
        self.level.entities.append(self.player)
        self.step_counter = 0
        return self.get_obs(), self.get_info()
    
    def get_obs(self):
        """
        Convert internal state to observation format. The observation includes:
        - position of the player
        - position of the enemy/enemies
        - position of the key
        - position of the door
        - current score
        - current reward
        """
        obs = {}
        # Add entity info
        for entity in self.level.entities:
            if entity._entity_id.value == EntityID.PLAYER.value:
                player_pos = np.array([entity.x, entity.y], dtype=float)
                obs["player"] = player_pos
            elif entity._entity_id.value == EntityID.GROUND_ENEMY.value:
                enemy_pos = np.array([entity.x, entity.y], dtype=float)
                obs["enemy"] = enemy_pos
            elif entity._entity_id.value == EntityID.KEY.value:
                key_pos = np.array([entity.x, entity.y], dtype=float)
                obs["key"] = key_pos
            elif entity._entity_id.value == EntityID.DOOR.value:
                door_pos = np.array([entity.x, entity.y], dtype=float)
                obs["door"] = door_pos
            else:
                raise ValueError(f"Unknown entity id: {entity._entity_id}")
        # Add current score
        obs["score"] = self.score
        # Add current reward
        obs["reward"] = self.level.reward
        return obs
    
    def get_info(self):
        """
        Compute auxiliary information.

        Returns:
            dict: Empty dict for now.
        """
        return {}


