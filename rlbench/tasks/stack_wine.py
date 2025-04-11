from typing import List, Tuple

import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped


class StackWine(Task):

    def init_task(self):
        wine_bottle = Shape('wine_bottle')
        self.tool_visual_name = 'wine_bottle'
        self.tool_name = ['wine_bottle', 'Panda_leftfinger_force_contact', 'Panda_rightfinger_force_contact']
        self.target_name = []
        self.distractor_names = []
        self.tool = [Shape(tool) for tool in self.tool_name]
        self.target = [Shape(tool) for tool in self.target_name]
        self.distractors = [Shape(tool) for tool in self.distractor_names]


        self.register_graspable_objects([wine_bottle])
        self.register_success_conditions(
            [DetectedCondition(wine_bottle, ProximitySensor('success'))])

    def init_episode(self, index: int) -> List[str]:
        return ['stack wine bottle',
                'slide the bottle onto the wine rack',
                'put the wine away',
                'leave the wine on the shelf',
                'grasp the bottle and put it away',
                'place the wine bottle on the wine rack']

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, -np.pi / 4.], [0, 0, np.pi / 4.]
