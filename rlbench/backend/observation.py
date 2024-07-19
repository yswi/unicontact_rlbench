import numpy as np


class Observation(object):
    """Storage for both visual and low-dimensional observations."""

    def __init__(self,
                 left_shoulder_rgb: np.ndarray,
                 left_shoulder_depth: np.ndarray,
                 left_shoulder_mask: np.ndarray,
                 left_shoulder_point_cloud: np.ndarray,
                 right_shoulder_rgb: np.ndarray,
                 right_shoulder_depth: np.ndarray,
                 right_shoulder_mask: np.ndarray,
                 right_shoulder_point_cloud: np.ndarray,
                 overhead_rgb: np.ndarray,
                 overhead_depth: np.ndarray,
                 overhead_mask: np.ndarray,
                 overhead_point_cloud: np.ndarray,
                 wrist_rgb: np.ndarray,
                 wrist_depth: np.ndarray,
                 wrist_mask: np.ndarray,
                 wrist_point_cloud: np.ndarray,
                 front_rgb: np.ndarray,
                 front_depth: np.ndarray,
                 front_mask: np.ndarray,
                 front_point_cloud: np.ndarray,
                 joint_velocities: np.ndarray,
                 joint_positions: np.ndarray,
                 joint_forces: np.ndarray,
                 gripper_open: float,
                 gripper_pose: np.ndarray,
                 gripper_matrix: np.ndarray,
                 gripper_joint_positions: np.ndarray,
                 gripper_touch_forces: np.ndarray,
                 task_low_dim_state: np.ndarray,
                 misc: dict,
                 contact_info: dict = None,
                 contact_est: np.ndarray = None,
                 ee_velocity: np.ndarray = None,
                 object_poses: dict = None,
                 keyframe: bool = False
                 ):
        self.left_shoulder_rgb = left_shoulder_rgb
        self.left_shoulder_depth = left_shoulder_depth
        self.left_shoulder_mask = left_shoulder_mask
        self.left_shoulder_point_cloud = left_shoulder_point_cloud
        self.right_shoulder_rgb = right_shoulder_rgb
        self.right_shoulder_depth = right_shoulder_depth
        self.right_shoulder_mask = right_shoulder_mask
        self.right_shoulder_point_cloud = right_shoulder_point_cloud
        self.overhead_rgb = overhead_rgb
        self.overhead_depth = overhead_depth
        self.overhead_mask = overhead_mask
        self.overhead_point_cloud = overhead_point_cloud
        self.wrist_rgb = wrist_rgb
        self.wrist_depth = wrist_depth
        self.wrist_mask = wrist_mask
        self.wrist_point_cloud = wrist_point_cloud
        self.front_rgb = front_rgb
        self.front_depth = front_depth
        self.front_mask = front_mask
        self.front_point_cloud = front_point_cloud
        self.joint_velocities = joint_velocities
        self.joint_positions = joint_positions
        self.joint_forces = joint_forces
        self.gripper_open = gripper_open
        self.gripper_pose = gripper_pose
        self.gripper_matrix = gripper_matrix
        self.gripper_joint_positions = gripper_joint_positions
        self.gripper_touch_forces = gripper_touch_forces
        self.task_low_dim_state = task_low_dim_state
        self.misc = misc
        self.contact_est = contact_est
        self.contact_info_raw = contact_info
        self.contact_info = self.refine_contact_info(contact_info)
        self.ee_velocity = ee_velocity
        self.contact_goal = None
        self.ignore_collision = False  # For perceiver-actor baseline
        self.object_poses = object_poses
        self.keyframe = keyframe

    def refine_contact_info(self, contact_info):
        contact_info_refined = {}

        # return empty dictionary if no contact.
        if contact_info is None:
            return contact_info_refined
        
        for source_object_candidate, contact_info_i in contact_info["contact_info"].items():
            cnt_pts = []
            cnt_force = []
            cnt_normal = []
            cnt_handles = []

            for i, cnt in enumerate(contact_info_i):
                if cnt['contact_handles'][1] not in contact_info['distractor_labels']:
                    cnt_pts.append(cnt['contact'][:3])
                    cnt_force.append(cnt['contact'][3:6])
                    cnt_normal.append(cnt['contact'][6:9])
                    cnt_handles.append(cnt['contact_handles'])

            contact_info_refined[source_object_candidate] = {
                'points_estimated': self.contact_est,
                'points': np.array(cnt_pts),
                'forces': np.array(cnt_force),
                'normals': np.array(cnt_normal),
                "cnt_handles": np.array(cnt_handles),
                "relative_pose": contact_info["relative_pose"],
                "labels": contact_info["labels"],
            }
        return contact_info_refined

    def get_low_dim_data(self) -> np.ndarray:
        """Gets a 1D array of all the low-dimensional observations.

        :return: 1D array of observations.
        """
        low_dim_data = [] if self.gripper_open is None else [[self.gripper_open]]
        for data in [self.joint_velocities, self.joint_positions,
                     self.joint_forces,
                     self.gripper_pose, self.gripper_joint_positions,
                     self.gripper_touch_forces, self.task_low_dim_state]:
            if data is not None:
                low_dim_data.append(data)
        return np.concatenate(low_dim_data) if len(low_dim_data) > 0 else np.array([])
