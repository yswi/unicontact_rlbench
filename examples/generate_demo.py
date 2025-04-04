import numpy as np
import os
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *
from rlbench.backend.conditions import DetectedCondition

from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
import rlbench.utils as utils
from rlbench.dataset_generator import save_demo
from rlbench.backend.const import *
from rlbench.backend.spawn_boundary import SpawnBoundary
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# TASKCONFIG = TaskConfig()
DIRT_NUM = 5



if __name__ == '__main__':
    # To use 'saved' demos, set the path below, and set live_demos=False
    live_demos = True
    DATASET = '' if live_demos else 'PATH/TO/YOUR/DATASET'

    obs_config = ObservationConfig()
    obs_config.set_all(True)

    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
        obs_config=ObservationConfig(),
        headless=False)
    scene_data = env.get_scene_data()

    env.launch()
    # task = env.get_task(SweepToDustpan)
    task = env.get_task(PlaceHangerOnRack)


    for i in range (5):
        SAVE_PATH = os.path.abspath( os.path.join(os.path.dirname(__file__), '../../dataset/'))
        variation_path = os.path.join(SAVE_PATH, task.get_name(), VARIATIONS_ALL_FOLDER)
        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        if not os.path.exists(episodes_path):
            os.makedirs(episodes_path)

        episode_path = os.path.join(episodes_path, EPISODE_FOLDER % i)

        demos = task.get_demos(1, live_demos=live_demos)  # -> List[List[Observation]]

        my_variation_count = -1 # TODO: make it variable
        demos.variation_number = my_variation_count
        save_demo(scene_data, demos, episode_path, my_variation_count)


        print("saved")
        np.save('demo_sweep_dustpan_%d.npy' % i, demos)


    print('Done')
    env.shutdown()
