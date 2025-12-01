import random
import time
import numpy as np
import os
import os.path as op
import sys

from env_src.getout.imageviewer import ImageViewer

# Set up path for importing getout
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from env_src.getout.getout.paramLevelGenerator import ParameterizedLevelGenerator
from env_src.getout.getout.getout import Getout
from env_src.getout.getout.actions import GetoutActions

# Stable Baselines PPO
from stable_baselines3 import PPO


# ------------------------------
# 1. Image viewer setup
# ------------------------------
def setup_image_viewer(getout):
    viewer = ImageViewer(
        "Getout PPO Demo",
        getout.camera.height,
        getout.camera.width,
        monitor_keyboard=True,
        relevant_keys=set([]),  # no keyboard control needed
    )
    return viewer


# ------------------------------
# 2. Create a Getout instance
# ------------------------------
def create_getout_instance():
    seed = random.random()
    env = Getout(start_on_first_action=True)

    level_generator = ParameterizedLevelGenerator()
    level_generator.generate(env, seed=seed)
    env.render()

    return env


# ------------------------------
# 3. Gym-style wrapper
# ------------------------------
class GetoutWrapper:
    """
    A simple wrapper giving the Getout env a Gym-like API.
    """

    def __init__(self):
        self.env = create_getout_instance()

    def reset(self):
        # If your env has a reset, use it; if not, recreate it
        self.env = create_getout_instance()
        return self._get_obs()

    def _get_obs(self):
        # Observation is the rendered camera image
        return self.env.get_obs()

    def step(self, action):
        """
        Action must be a list of the Getout action integers.
        PPO usually outputs a discrete integer.

        If your PPO agent outputs a single int, convert it.
        """
        if isinstance(action, np.ndarray):
            action = action.item()

        # Convert agent action → environment list
        action_list = []

        if action == 0:
            pass # IDLE
        elif action == 1:
            action_list.append(GetoutActions.MOVE_LEFT.value)
        elif action == 2:
            action_list.append(GetoutActions.MOVE_RIGHT.value)
        elif action == 3:
            action_list.append(GetoutActions.MOVE_UP.value)
        else:
            # No-op action
            pass

        obs, reward, terminated, truncated, info = self.env.step(action_list)
        done = terminated or truncated
        return obs, reward, done, {}


# ------------------------------
# 4. Run PPO agent with viewer
# ------------------------------
def run_agent(model_path):
    # Create wrapped env
    env = GetoutWrapper()

    # Load PPO model
    print("Loading PPO agent:", model_path)
    agent = PPO.load(model_path)

    # Create viewer for visualization
    viewer = setup_image_viewer(env.env)

    np_img = np.asarray(env.env.camera.screen)
    viewer.show(np_img[:, :, :3])

    obs = env.reset()

    fps = 10
    target_frame_duration = 1 / fps
    last_frame_time = 0

    time.sleep(10)  # Wait 10 seconds before starting

    while True:
        current_frame_time = time.time()

        # Framerate control
        if last_frame_time + target_frame_duration > current_frame_time:
            time.sleep((last_frame_time + target_frame_duration) - current_frame_time)
            continue
        last_frame_time = current_frame_time

        # Agent selects action
        action, _ = agent.predict(obs, deterministic=True)

        # Step environment
        obs, reward, done, info = env.step(action)
        # Render env
        env.env.render()

        # Render image
        np_img = np.asarray(env.env.camera.screen)
        viewer.show(np_img[:, :, :3])

        if done or viewer.is_escape_pressed:
            break

    print("PPO agent visualization finished.")


if __name__ == "__main__":
    model_to_load_path = op.join("evaluate_results", "agent_ppo_custom", "checkpoints_2", "ppo_getout_700000_steps.zip")
    run_agent(model_to_load_path)
