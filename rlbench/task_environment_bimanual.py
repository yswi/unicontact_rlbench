import logging
from typing import List, Callable

import numpy as np
from pyrep import PyRep
from pyrep.const import ObjectType
from rlbench import utils
from rlbench.action_modes.action_mode import ActionMode
from rlbench.backend.exceptions import BoundaryError, WaypointError, \
    TaskEnvironmentError
from rlbench.backend.observation import Observation
from rlbench.backend.robot import Robot
from rlbench.backend.scene import Scene
from rlbench.backend.task import Task
from rlbench.demo import Demo
from rlbench.observation_config import ObservationConfig

_DT = 0.05
_MAX_RESET_ATTEMPTS = 40
_MAX_DEMO_ATTEMPTS = 10


class TaskEnvironmentBimanual(object):

    def __init__(self,
                 pyrep: PyRep,
                 robots,
                 scene: Scene,
                 task: Task,
                 action_mode: ActionMode,
                 dataset_root: str,
                 obs_config: ObservationConfig,
                 static_positions: bool = False,
                 attach_grasped_objects: bool = True,
                 shaped_rewards: bool = False
                 ):
        self._pyrep = pyrep
        self._left_robot = robots[0]
        self._right_robot = robots[1]

        self._scene = scene
        self._task = task
        self._variation_number = 0
        self._action_mode = action_mode
        self._dataset_root = dataset_root
        self._obs_config = obs_config
        self._static_positions = static_positions
        self._attach_grasped_objects = attach_grasped_objects
        self._shaped_rewards = shaped_rewards
        self._reset_called = False
        self._prev_ee_velocity = None
        self._enable_path_observations = False

        self._scene.load(self._task)
        self._pyrep.start()
        self._left_robot_shapes = self._left_robot.arm.get_objects_in_tree(
            object_type=ObjectType.SHAPE)
        self._right_robot_shapes = self._right_robot.arm.get_objects_in_tree(
            object_type=ObjectType.SHAPE)

    def get_name(self) -> str:
        return self._task.get_name()

    def sample_variation(self) -> int:
        self._variation_number = np.random.randint(
            0, self._task.variation_count())
        return self._variation_number

    def set_variation(self, v: int) -> None:
        if v >= self.variation_count():
            raise TaskEnvironmentError(
                'Requested variation %d, but there are only %d variations.' % (
                    v, self.variation_count()))
        self._variation_number = v

    def variation_count(self) -> int:
        return self._task.variation_count()

    def reset(self) -> (List[str], Observation):
        self._scene.reset()
        try:
            desc = self._scene.init_episode(
                self._variation_number, max_attempts=_MAX_RESET_ATTEMPTS,
                randomly_place=not self._static_positions)
        except (BoundaryError, WaypointError) as e:
            raise TaskEnvironmentError(
                'Could not place the task %s in the scene. This should not '
                'happen, please raise an issues on this task.'
                % self._task.get_name()) from e

        self._reset_called = True
        # Returns a list of descriptions and the first observation
        return desc, self._scene.get_observation()

    def get_observation(self) -> Observation:
        return self._scene.get_observation()

    # TODO: track step function to use the desired arm idx
    def step(self, action, arm_idx) -> (Observation, int, bool):
        # returns observation, reward, done, info
        if not self._reset_called:
            raise RuntimeError(
                "Call 'reset' before calling 'step' on a task.")
        # Assign which arm to move 
        self._scene.robot = self._scene.left_robot if arm_idx == 0 else self._scene.right_robot
        self._action_mode[arm_idx].action(self._scene, action)
        success, terminate = self._task.success()
        reward = float(success)
        if self._shaped_rewards:
            reward = self._task.reward()
            if reward is None:
                raise RuntimeError(
                    'User requested shaped rewards, but task %s does not have '
                    'a defined reward() function.' % self._task.get_name())
        return self._scene.get_observation(), reward, terminate

    def get_demos(self, amount: int, live_demos: bool = False,
                  image_paths: bool = False,
                  callable_each_step: Callable[[Observation], None] = None,
                  max_attempts: int = _MAX_DEMO_ATTEMPTS,
                  random_selection: bool = True,
                  from_episode_number: int = 0,
                  ) -> List[Demo]:
        """Negative means all demos"""
        if not live_demos and (self._dataset_root is None
                               or len(self._dataset_root) == 0):
            raise RuntimeError(
                "Can't ask for a stored demo when no dataset root provided.")

        if not live_demos:
            if self._dataset_root is None or len(self._dataset_root) == 0:
                raise RuntimeError(
                    "Can't ask for stored demo when no dataset root provided.")
            demos = utils.get_stored_demos(
                amount, image_paths, self._dataset_root, self._variation_number,
                self._task.get_name(), self._obs_config,
                random_selection, from_episode_number)
        else:
            left_ctr_loop = self._left_robot.arm.joints[0].is_control_loop_enabled()
            right_ctr_loop = self._left_robot.arm.joints[0].is_control_loop_enabled()

            self._left_robot.arm.set_control_loop_enabled(True)
            self._right_robot.arm.set_control_loop_enabled(True)

            demos = self._get_live_demos_(amount, callable_each_step, max_attempts)
            self._left_robot.arm.set_control_loop_enabled(left_ctr_loop)
            self._right_robot.arm.set_control_loop_enabled(left_ctr_loop)
      
        return demos



    def _get_live_demos_(self, amount: int,
                        callable_each_step: Callable[
                            [Observation], None] = None,
                        max_attempts: int = _MAX_DEMO_ATTEMPTS) -> List[Demo]:
        ## TODO : Integrate this with l4c_rlbench

        demos = []
        # for i in range(amount):
        attempts = max_attempts
        # print( 'Getting demo %d/%d' % (i + 1, amount))
        while attempts > 0:
            random_seed = np.random.get_state()
            self.reset()
            for s in self._pyrep.get_objects_in_tree():
                try:
                    if s.get_name() == 'diningTable':
                        target = s
                except:
                    pass
            # demo = self._scene.get_demo(callable_each_step=callable_each_step,
            #                             contact=(self._task.tool, target))
            try:
                print("start demo")
                demo = self._scene.get_demo(callable_each_step=callable_each_step,
                                            contact=(self._task.tool, target))
                demo.random_seed = random_seed
                print("success and return" , demo)
                return demo
            except:
                attempts += 1
            #     print("demo failed")
                # demo = self._scene.get_demo( callable_each_step=callable_each_step)
                # demo.random_seed = random_seed
                # demos.append(demo)

            #         # break
            #     # except Exception as e:
            #     #     attempts -= 1
            #     #     logging.info('Bad demo. ' + str(e))
            # if attempts <= 0:
            #     raise RuntimeError(
            #         'Could not collect demos. Maybe a problem with the task?')


    def reset_to_demo(self, demo: Demo) -> (List[str], Observation):
        demo.restore_state()
        return self.reset()