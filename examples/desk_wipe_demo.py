import numpy as np

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import WipeDesk
from typing import List

from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import EmptyCondition
from rlbench.backend.spawn_boundary import SpawnBoundary
# import l4c_rlbench.utils.utils as utils
# from l4c_rlbench.rigid_dynamics import RigidDynamics
# from l4c_rlbench.cfg.config import MppiConfig
# from language4contact.config.task_policy_configs import TaskConfig

# TASKCONFIG = TaskConfig()

def wipe_desk_init_task(Task: Task):
    def init_task_(Task: Task, *args, **kwarg):
        Task.dirt_spots = []
        Task.tool_name = 'sponge'
        # print(grasp_target_name, kwarg)
        Task.tool_name = kwarg['tool_name'] if 'tool_name' in kwarg.keys() else 'sponge'
        Task.target_name = kwarg['target_name'] if 'target_name' in kwarg.keys() else 'diningTable'
        Task.grasp_target_name = 'sponge'
        # Task.grasp_target_name = kwarg['grasp_target_name'] if 'grasp_target_name' in kwarg.keys() else Task.tool_name
        print("grasp target name", Task.grasp_target_name)
        Task.tool = Shape(Task.tool_name)

        Task.grasp_target = Shape(Task.grasp_target_name)
        Task.sensor = ProximitySensor('sponge_sensor')
        Task.register_graspable_objects([Task.tool])

        boundaries = [Shape('dirt_boundary')]
        _, _, Task.z_boundary = boundaries[0].get_position()
        Task.b = SpawnBoundary(boundaries)
        Task.target = Shape(Task.target_name)

        Task.contact_goal = []
        Task.tool_pcd = utils.get_tool_o3d(Task, N = 10000)
        Task.handle_pcd= utils.get_handle_o3d(Task)

        Task.tool_dynamics = RigidDynamics # (Task.tool, Task.target, cfg=MppiConfig())
        # Task.tool.set_model_renderable(False)

        try:
            Task.tool_visual = Shape('sponge_visual')
            Task.render = [Task.tool.set_model_renderable,
                           Task.tool.set_renderable,
                           Task.tool_visual.set_model_renderable,
                           Task.tool_visual.set_renderable
                           ]
        except:
            Task.render = [Task.tool.set_model_renderable,
                           Task.tool.set_renderable]

        for r in Task.render:
            r(False)


        return ['wipe the dots up',
                'use the sponge to clean up the dirt',
                'wipe dirt off the desk',
                'use the sponge to clean up the desk',
                'remove the dirt from the desk',
                'grip the sponge and wipe it back and forth over any dirt you see',
                'clean up the mess',
                'wipe the dirt up']
        # b = SpawnBoundary([Shape('workspace0')])
        # b.sample(Task.tool, ignore_collisions=True, min_distance=0.1)
        # Task.tool.set_orientation([0, 0, -np.pi/2], relative_to=Shape('workspace'), reset_dynamics=False)


    return init_task_

def check(task, tot_tick, goal_steps):
    completed = False
    if len(task.dirt_spots) == 0:
        completed = True
    # if tot_tick > 80:
    #     completed = True

    if goal_steps > 20:
        completed = True
    return completed

def final_action(obs):
    return np.array([0, 0, 0, 0, 0, 0])


class WipeDesk(WipeDesk):
    @wipe_desk_init_task
    def init_task(self, **kwarg) -> None:

        self.grasp_target_name = None
        pass
    def get_completed(self,tot_tick, goal_steps):
        return check(self, tot_tick, goal_steps)
    def get_final_action(self, obs):
        return final_action(obs)
    def subgoal_success_threshold(self):
        return 0.23 #0.13

    def get_score(self,t):
        return 100 - len(self.dirt_spots)

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
    env.launch()

    task = env.get_task(WipeDesk)

    '''
    Open-loop imitation learning
    '''
    for i in range (0, 10):
        demos = task.get_demos(1, live_demos=live_demos)  # -> List[List[Observation]]
        # try:
        #     print("start demo")
        #     demos = task.get_demos(1, live_demos=live_demos)  # -> List[List[Observation]]
        #     print("end demo")
        #     demos = np.array(demos).flatten()
        #     np.save('demo_wipe_desk_%d.npy' % i, demos)

        #     print("saved")
        # except:
        #     print("failed")

    print('Done')
    env.shutdown()
