import numpy as np
import os
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import SweepToDustpan
from rlbench.backend.conditions import DetectedCondition

from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
import rlbench.utils as utils
from rlbench.dataset_generator import save_demo
from rlbench.backend.const import *
# from l4c_rlbench.rigid_dynamics import RigidDynamics
# from l4c_rlbench.cfg.config import MppiConfig
# from language4contact.config.task_policy_configs import TaskConfig
from rlbench.backend.spawn_boundary import SpawnBoundary
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# TASKCONFIG = TaskConfig()
DIRT_NUM = 5



def sweep_init_task(Task: Task) -> None:
    def init_task_(Task: Task, **kwarg):
        if 'grasp_target_name' in  kwarg.keys():
            Task.grasp_target_name = kwarg['grasp_target_name']

        Task.dirt_spots = []
        # print(grasp_target_name, kwarg)

        Task.tool_visual_name = 'sweep_to_dustpan_broom_visual'

        Task.tool_name = 'broom'
        Task.target_name = kwarg['target_name'] if 'target_name' in kwarg.keys() else 'diningTable'
        Task.grasp_target_name = 'broom_handle'

        Task.tool = Shape(Task.tool_name)
        Task.grasp_target = Shape(Task.grasp_target_name)

        success_sensor = ProximitySensor('success')
        dirts = [Shape('dirt' + str(i)) for i in range(DIRT_NUM)]
        conditions = [DetectedCondition(dirt, success_sensor) for dirt in dirts]
        Task.register_graspable_objects([Task.tool])
        Task.register_success_conditions(conditions)


        Task.target = Shape(Task.target_name)

        Task.contact_goal = []
        Task.tool_pcd = utils.get_tool_o3d(Shape(Task.tool_visual_name), N = 20000)
        Task.handle_pcd= utils.get_handle_o3d(Task)


        Task.non_collision_model = Shape('Dustpan_4')
        Task.tool.set_model_renderable(False)
        Task.non_collision_model.set_model_renderable(False)

        import open3d as o3d
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(Task.tool_pcd)
        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        #     size=0.6, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries(
        #     [ pcd_o3d, mesh_frame])

        # Task.grasp_target.set_model_detectable(False)
        # Task.grasp_target.set_model_measurable(False)

        # Task.grasp_target.set_renderable(False)
        # Task.grasp_target.set_detectable(False)

        # Task.tool_dynamics = RigidDynamics(Task.tool, Task.target, cfg=MppiConfig())



    return init_task_

def final_action(obs):
    return np.array([0, 0, 0.1, 0, 0, 0])

def check(task, tot_tick, goal_steps):
    completed = False
    if goal_steps > 3:
        completed = True
    if tot_tick > 50:
        completed = True
    return completed




class SweepToDustpan(SweepToDustpan):
    def init_task(self) -> None:
        self.dirt_spots = []
        self.tool_visual_name = 'broom'
        # self.tool_name = ['broom', 'Panda_leftfinger_respondable'] # tool or gripper
        self.tool_name = ['broom', 'Panda_leftfinger_force_contact'] # tool or gripper
        self.target_name = 'diningTable'
        self.distractor_names = ['broom_holder', 'Panda_rightfinger_respondable']
        self.grasp_target_name = 'broom_handle'

        self.tool = [ Shape(tool) for tool in self.tool_name ]
        self.grasp_target = Shape(self.grasp_target_name)
        self.distractors = [ Shape(distractor) for distractor in self.distractor_names ]


        self.success_sensor = ProximitySensor('success')
        self.dirts = [Shape('dirt' + str(i)) for i in range(DIRT_NUM)]
        conditions = [DetectedCondition(dirt, self.success_sensor) for dirt in self.dirts]
        self.register_graspable_objects([self.tool[0]]) # Otherwise grasping is not stable. (slips)
        self.register_success_conditions(conditions)

        # print("rearrange broom")
        # tool_orientation = self.broom.get_quaternion()
        # tool_orientation_new = R.from_quat(tool_orientation).as_matrix() @ R.from_euler('xyz', [0, 0, -np.pi/2]).as_matrix() 
        # self.broom.set_quaternion(R.from_matrix(tool_orientation_new).as_quat())

        # tool_orientation = self.tool[0].get_quaternion()
        # tool_orientation_new = R.from_quat(tool_orientation) @ R.from_euler('xyz', [0, 0, np.pi/2])
        # self.tool[0].set_quaternion(tool_orientation_new.as_quat())

        self.target = None #Shape(self.target_name)

        self.contact_goal = []
        self.tool_pcd = utils.get_tool_o3d(Shape(self.tool_visual_name), N=10000)
        self.handle_pcd = utils.get_handle_o3d(self)

        self.non_collision_model = Shape('Dustpan_4')
        # self.render = [self.tool.set_model_renderable,
        #                self.tool.set_renderable,
        #                self.non_collision_model.set_model_renderable,
        #                self.non_collision_model.set_renderable]

        # for r in self.render:
        #     r(False)

        self.score = [False for _ in range(DIRT_NUM)]
        # pcd_o3d = o3d.geometry.PointCloud()
        # pcd_o3d.points = o3d.utility.Vector3dVector(self.tool_pcd)
        # o3d.io.write_point_cloud("broom.ply", pcd_o3d)


    def subgoal_success_threshold(self):
        return 0.1


def get_completed(self, tot_tick, goal_steps):
    return check(self, tot_tick, goal_steps)


def get_final_action(self, obs):
    return final_action(obs)


def subgoal_success_threshold(self):
    return 0.1


def get_score(self, t):
    score = [DetectedCondition(dirt, self.success_sensor).condition_met()[0] for dirt in self.dirts]
    for i in range(DIRT_NUM):
        if not self.score[i]:
            self.score[i] = score[i]
    return np.sum(self.score) / len(self.score) * 100


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
    task = env.get_task(SweepToDustpan)


    for i in range (110):
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
