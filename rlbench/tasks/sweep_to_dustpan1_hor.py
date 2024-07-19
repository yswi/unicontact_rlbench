from typing import List
import numpy as np

from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition
from rlbench import utils
from rlbench.backend.spawn_boundary import SpawnBoundary

DIRT_NUM = 5


class SweepToDustpan1Hor(Task):

    def init_task(self) -> None:

        self.dirt_spots = []
        # print(grasp_target_name, kwarg)

        self.tool_visual_name = 'broom'

        self.tool_name = 'broom'
        self.target_name = 'diningTable'
        self.grasp_target_name = 'broom_handle'

        self.tool = Shape(self.tool_name)
        self.grasp_target = Shape(self.grasp_target_name)

        self.success_sensor = ProximitySensor('success')
        self.dirts = [Shape('dirt' + str(i)) for i in range(DIRT_NUM)]
        conditions = [DetectedCondition(dirt, self.success_sensor) for dirt in self.dirts]
        self.register_graspable_objects([self.tool])
        self.register_success_conditions(conditions)

        self.target = Shape(self.target_name)

        self.contact_goal = []
        self.tool_pcd = utils.get_tool_o3d(Shape(self.tool_visual_name), N=10000)
        self.handle_pcd = utils.get_handle_o3d(self)

        self.non_collision_model = Shape('Dustpan_4')
        self.render = [self.tool.set_model_renderable,
                       self.tool.set_renderable,
                       self.non_collision_model.set_model_renderable,
                       self.non_collision_model.set_renderable]
        b = SpawnBoundary([Shape('workspace0')])

        b.sample(self.tool, ignore_collisions=True, min_distance=0.05)

        for r in self.render:
            r(False)
            # r(True)
        self.score = [False for _ in range(DIRT_NUM)]

        import open3d as o3d
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(self.tool_pcd)

    def get_completed(self, tot_tick, goal_steps):
        return check(self, tot_tick, goal_steps)

    def get_final_action(self, obs):
        return final_action(obs)

    def subgoal_success_threshold(self):
        return 0.3 # 0.1

    def get_score(self, t):
        score = [DetectedCondition(dirt, self.success_sensor).condition_met()[0] for dirt in self.dirts]
        for i in range(DIRT_NUM):
            if not self.score[i]:
                self.score[i] = score[i]

        # print([DetectedCondition(dirt, self.success_sensor).condition_met()[0] for dirt in self.dirts])
        return np.sum(self.score) / len(self.score) * 100

    def init_episode(self, index: int) -> List[str]:
        return ['sweep dirt to dustpan',
                'sweep the dirt up',
                'use the broom to brush the dirt into the dustpan',
                'clean up the dirt',
                'pick up the brush and clean up the table',
                'grasping the broom by its handle, clear way the dirt from the '
                'table',
                'leave the table clean']

    def variation_count(self) -> int:
        return 1
def check(task, tot_tick, goal_steps):
    completed = False
    if goal_steps > 3:
        completed = True
    if tot_tick > 50:
        completed = True
    return completed
