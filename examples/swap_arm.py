import numpy as np

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
from rlbench.tasks import SweepToDustpan
from pyrep.objects.dummy import Dummy

class Agent(object):

    def __init__(self, action_shape):
        self.action_shape = action_shape

    def act(self, obs):
        # arm = np.random.normal(0.0, 0.1, size=(self.action_shape[0] - 1,))
        arm = np.ones((self.action_shape[0] - 1,)) * 0.02
        gripper = [0.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)


obs_config = ObservationConfig()
obs_config.set_all(True)
obs_config.gripper_touch_forces = False

action_mode = MoveArmThenGripper(
    arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
env = Environment(
    action_mode, obs_config=obs_config, headless=False,
    robot_setup='baxter')
env.launch()

task = env.get_task(SweepToDustpan)


waypoints = [Dummy('waypoint%d' % i) for i in range(1)]

path = env._robots[1].arm.get_path(position=np.array([0.17500001, 0.12499997 ,0.9346686 ]) ,
                            quaternion=np.array([0.70711589, 0.,0.,0.70709765]))
done = False
while not done:
    done = path.step()
    pr.step()


# agent = Agent(env.action_shape)  # 6DoF + 1 for gripper

# training_steps = 200
# episode_length = 200
# obs = None

# for idx in range(2):
#     for i in range(training_steps):
#         if i % episode_length == 0:
#             print('Reset Episode')
#             descriptions, obs = task.reset()
#             print(descriptions)
#         action = agent.act(obs)
#         print(action)
#         obs, reward, terminate = task.step(action, arm_idx = idx)

print('Done')
env.shutdown()
