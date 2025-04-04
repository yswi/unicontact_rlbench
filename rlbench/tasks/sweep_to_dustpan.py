from typing import List
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition
import rlbench.utils as utils
import numpy as np
from scipy.spatial.transform import Rotation as R

DIRT_NUM = 5

class SweepToDustpan(Task):

    def init_task(self) -> None:
        self.dirt_spots = []
        self.tool_visual_name = 'broom'
        self.tool_name = ['broom', 'Panda_leftfinger_force_contact', 'Panda_rightfinger_respondable']
        self.target_name = 'diningTable'
        self.distractor_names = ['broom_holder']
        self.grasp_target_name = 'broom_handle'

        self.tool = [Shape(tool) for tool in self.tool_name]
        self.grasp_target = Shape(self.grasp_target_name)
        self.distractors = [Shape(distractor) for distractor in self.distractor_names]

        self.success_sensor = ProximitySensor('success')
        self.dirts = [Shape('dirt' + str(i)) for i in range(DIRT_NUM)]
        conditions = [DetectedCondition(dirt, self.success_sensor) for dirt in self.dirts]
        self.register_graspable_objects([self.tool[0]])
        self.register_success_conditions(conditions)

        self.target = None

        self.contact_goal = []
        self.tool_pcd = utils.get_tool_o3d(Shape(self.tool_visual_name), N=10000)
        self.handle_pcd = utils.get_handle_o3d(self)

        self.non_collision_model = Shape('Dustpan_4')

        self.score = [False for _ in range(DIRT_NUM)]

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
    
    def subgoal_success_threshold(self):
        return 0.1

    def get_completed(self, tot_tick, goal_steps):
        completed = False
        if goal_steps > 3:
            completed = True
        if tot_tick > 50:
            completed = True
        return completed

    def get_final_action(self, obs):
        return np.array([0, 0, 0.1, 0, 0, 0])

    def get_score(self, t):
        score = [DetectedCondition(dirt, self.success_sensor).condition_met()[0] for dirt in self.dirts]
        for i in range(DIRT_NUM):
            if not self.score[i]:
                self.score[i] = score[i]
        return np.sum(self.score) / len(self.score) * 100
