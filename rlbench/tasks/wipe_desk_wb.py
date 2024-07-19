from typing import List
from pyrep.const import PrimitiveShape
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import EmptyCondition
from rlbench.backend.spawn_boundary import SpawnBoundary
import random
DIRT_POINTS = 100


class WipeDeskWb(Task):

    def init_task(self) -> None:
        self.dirt_spots = []
        self.tool = Shape('sponge')
        self.sensor = ProximitySensor('sponge_sensor')
        self.register_graspable_objects([self.tool])

        boundaries = [Shape('dirt_boundary')]
        _, _, self.z_boundary = boundaries[0].get_position()
        self.b = SpawnBoundary(boundaries)

    def init_episode(self, index: int) -> List[str]:
        self._place_dirt()
        self.register_success_conditions([EmptyCondition(self.dirt_spots)])
        return ['wipe the dots up',
                'use the eraser to clean up the dirt',
                'use the sponge to clean up the desk',
                'remove the dirt from the desk',
                'grip the sponge and wipe it back and forth over any dirt you see',
                'clean up the mess',
                'wipe the dirt up']

    def variation_count(self) -> int:
        return 1

    def step(self) -> None:
        for d in self.dirt_spots:
            if self.sensor.is_detected(d):
                self.dirt_spots.remove(d)
                d.remove()

    def cleanup(self) -> None:
        for d in self.dirt_spots:
            d.remove()
        self.dirt_spots = []

    def _place_dirt(self):
        self.cleanup()
        # color = [round(random.random()*1, 2),
        #            round(random.random()*1, 2),
        #            round(random.random()*1, 2)]
        for i in range(DIRT_POINTS):
            spot = Shape.create(type=PrimitiveShape.CUBOID,
                                size=[.006, .006, .001],
                                mass=0, static=True, respondable=False,
                                renderable=True,
                                color=[0., 0., 0.])
            spot.set_parent(self.get_base())
            spot.set_position([-1, -1, self.z_boundary + 0.001])
            self.b.sample(spot, min_distance=0.00,
                          min_rotation=(0.00, 0.00, 0.00),
                          max_rotation=(0.00, 0.00, 0.00))
            self.dirt_spots.append(spot)
        self.b.clear()
