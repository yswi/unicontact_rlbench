import importlib
import pickle
from os import listdir
from os.path import join, exists
from typing import List
import open3d as o3d

import numpy as np
from PIL import Image
from natsort import natsorted
from pyrep.objects import VisionSensor

from rlbench.backend.const import *
from rlbench.backend.utils import image_to_float_array, rgb_handles_to_mask
from rlbench.demo import Demo
from rlbench.observation_config import ObservationConfig
from scipy.spatial.transform import Rotation as R

def transform_mesh(v, pose, device, from_euler = False, return_numpy = False):
    '''
    :param mesh: mesh object
    :param pose: tensor [x,y,z,qx,qy,qz,qw] format
    :return: transformed mesh in tensor
    '''

    if from_euler:
        pose_ = pose[:, 3:] # TODO: Check this. Not sure why 3:-1
        a_r = R.from_euler('XYZ', pose_)

        # a_r = R.from_euler('zyx', pose_)
    else:
        pose_ = pose[:, 3:7]
        a_r = R.from_quat(pose_ ) # x,y,z,w format

    a_T = a_r.as_matrix()

    B = a_T.shape[0]
    T = np.zeros((B, 4, 4))  # B x 4 x 4
    T[:, :3, :3] =  a_T
    T[:, :3, 3] = pose[:, :3]
    T[:, 3, 3] = 1
    #
    # # TODO: object transformation is not consistent
    # T_off1 = np.zeros((B, 4, 4))  # B x 4 x 4
    # T_off1[:, :3, :3] = R.from_euler('xyz', [-2.3874e+01, +1.2650e-02, -8.9991e+01]).as_matrix()
    # T_off1[:, 3, 3] = 1
    # T_off1 = torch.tensor( np.linalg.inv(T_off1)).to(device)
    #
    # # TODO: object transformation is not consistent
    # T_off2 = np.zeros((B, 4, 4))  # B x 4 x 4
    # T_off2[:, :3, :3] = R.from_euler('xyz', [-np.pi/2,0 , 0]).as_matrix()
    # T_off2[:, 3, 3] = 1
    # T_off2 = torch.tensor( np.linalg.inv(T_off2)).to(device)

    # Current State.
    V = np.ones((T.shape[0], v.shape[0], 4))  # B x N x 4
    V[:, :, :3] = v[np.newaxis, :, :].repeat(T.shape[0], axis=0)  # B x N x 4

    ## Resultant mesh pointcloud.
    # T = torch.tensor(T).to(device)
    # V = torch.tensor(V).to(device)
    # V_ =  T @ V.transpose(1, 2)  # B x 4 x N
    # V_ = V_.transpose(1, 2)  # B x N x 4
    V_ =  T @ np.transpose(V, (0, 2, 1))  # B x 4 x N
    V_ = np.transpose(V_,(0, 2, 1))  # B x N x 4
    tool_pcd = V_[:, :, :3]

    # visualize with o3d with frame
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, 0])
    pcds = [mesh_frame]

    # if tool_pcd.shape[0] > 1:
    #     for i in range(10):
    #         pcd = o3d.geometry.PointCloud()
    #
    #         pcd.points = o3d.utility.Vector3dVector(tool_pcd[i].cpu().numpy())
    #         pcds.append(pcd)
    #     # print("visualize", pose_)
    #     o3d.visualization.draw_geometries(pcds)


    if return_numpy:
        return tool_pcd.cpu().numpy()
    return tool_pcd


class InvalidTaskName(Exception):
    pass


def get_handle_o3d(task):
    # Get next tool coords.
    vertices, indices, normals = task.grasp_target.get_mesh_data()
    # make new mesh from scrat
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(indices))
    mesh.triangle_normals = o3d.utility.Vector3dVector(normals)

    pcd = mesh.sample_points_uniformly(number_of_points=1000)
    return np.array(pcd.points)

def get_tool_o3d(task, N = 1000):
    # Get next tool coords.
    try:
        vertices, indices, normals = task.tool.get_mesh_data()
    except:
        vertices, indices, normals = task.get_mesh_data()
    # make new mesh from scrat
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(indices))
    mesh.triangle_normals = o3d.utility.Vector3dVector(normals)

    pcd = mesh.sample_points_uniformly(number_of_points=N)
    return np.array(pcd.points)

def name_to_task_class(task_file: str):
    name = task_file.replace('.py', '')
    class_name = ''.join([w[0].upper() + w[1:] for w in name.split('_')])
    mod = importlib.import_module("rlbench.tasks.%s" % name)
    mod = importlib.reload(mod)
    try:
        mod = importlib.import_module("rlbench.tasks.%s" % name)
        mod = importlib.reload(mod)
    except ModuleNotFoundError as e:
        raise InvalidTaskName(
            "The task file '%s' does not exist or cannot be compiled."
            % name) from e
    try:
        task_class = getattr(mod, class_name)
    except AttributeError as e:
        raise InvalidTaskName(
            "Cannot find the class name '%s' in the file '%s'."
            % (class_name, name)) from e
    return task_class


def get_stored_demos(amount: int, image_paths: bool, dataset_root: str,
                     variation_number: int, task_name: str,
                     obs_config: ObservationConfig,
                     random_selection: bool = True,
                     from_episode_number: int = 0) -> List[Demo]:

    task_root = join(dataset_root, task_name)
    if not exists(task_root):
        raise RuntimeError("Can't find the demos for %s at: %s" % (
            task_name, task_root))

    # Sample an amount of examples for the variation of this task
    if variation_number == -1:
        examples_path = join(
            task_root, 'all_variations',
            EPISODES_FOLDER)
    else:
        examples_path = join(
            task_root, VARIATIONS_FOLDER % variation_number,
            EPISODES_FOLDER)
    examples = listdir(examples_path)
    if amount == -1:
        amount = len(examples)
    if amount > len(examples):
        raise RuntimeError(
            'You asked for %d examples, but only %d were available.' % (
                amount, len(examples)))
    if random_selection:
        selected_examples = np.random.choice(examples, amount, replace=False)
    else:
        selected_examples = natsorted(
            examples)[from_episode_number:from_episode_number+amount]
    # print("getting ",from_episode_number, amount, selected_examples, task_root, examples)
    # Process these examples (e.g. loading observations)
    demos = []
    for example in selected_examples:
        example_path = join(examples_path, example)

        import sys
        from types import ModuleType
        def update_module_path_in_pickled_object(
                pickle_path: str, old_module_path: str, new_module: ModuleType
        ) -> None:
            """Update a python module's dotted path in a pickle dump if the
            corresponding file was renamed.

            Implements the advice in https://stackoverflow.com/a/2121918.

            Args:
                pickle_path (str): Path to the pickled object.
                old_module_path (str): The old.dotted.path.to.renamed.module.
                new_module (ModuleType): from new.location import module.
            """
            sys.modules[old_module_path] = new_module

            dic = pickle.load(open(pickle_path, "rb"))
            return dic
            # # dic = torch.load(pickle_path, map_location="cpu")
            #
            # del sys.modules[old_module_path]
            #
            # pickle.dump(dic, open(pickle_path, "wb"))
            # torch.save(dic, pickle_path)
        from rlbench.backend import observation
        obs = update_module_path_in_pickled_object(join(example_path, LOW_DIM_PICKLE), 'l4c_rlbench.rlbench.backend.observation', observation)

        # with open(join(example_path, LOW_DIM_PICKLE), 'rb') as f:
            # obs = pickle.load(f)

        l_sh_rgb_f = join(example_path, LEFT_SHOULDER_RGB_FOLDER)
        l_sh_depth_f = join(example_path, LEFT_SHOULDER_DEPTH_FOLDER)
        l_sh_mask_f = join(example_path, LEFT_SHOULDER_MASK_FOLDER)
        r_sh_rgb_f = join(example_path, RIGHT_SHOULDER_RGB_FOLDER)
        r_sh_depth_f = join(example_path, RIGHT_SHOULDER_DEPTH_FOLDER)
        r_sh_mask_f = join(example_path, RIGHT_SHOULDER_MASK_FOLDER)
        oh_rgb_f = join(example_path, OVERHEAD_RGB_FOLDER)
        oh_depth_f = join(example_path, OVERHEAD_DEPTH_FOLDER)
        oh_mask_f = join(example_path, OVERHEAD_MASK_FOLDER)
        wrist_rgb_f = join(example_path, WRIST_RGB_FOLDER)
        wrist_depth_f = join(example_path, WRIST_DEPTH_FOLDER)
        wrist_mask_f = join(example_path, WRIST_MASK_FOLDER)
        front_rgb_f = join(example_path, FRONT_RGB_FOLDER)
        front_depth_f = join(example_path, FRONT_DEPTH_FOLDER)
        front_mask_f = join(example_path, FRONT_MASK_FOLDER)

        num_steps = len(obs)

        # if not (num_steps == len(listdir(l_sh_rgb_f)) == len(
        #         listdir(l_sh_depth_f)) == len(listdir(r_sh_rgb_f)) == len(
        #         listdir(r_sh_depth_f)) == len(listdir(oh_rgb_f)) == len(
        #         listdir(oh_depth_f)) == len(listdir(wrist_rgb_f)) == len(
        #         listdir(wrist_depth_f)) == len(listdir(front_rgb_f)) == len(
        #         listdir(front_depth_f))):
        #     raise RuntimeError('Broken dataset assumption')

        for i in range(num_steps):
            si = IMAGE_FORMAT % i
            if obs_config.left_shoulder_camera.rgb:
                obs[i].left_shoulder_rgb = join(l_sh_rgb_f, si)
            if obs_config.left_shoulder_camera.depth or obs_config.left_shoulder_camera.point_cloud:
                obs[i].left_shoulder_depth = join(l_sh_depth_f, si)
            if obs_config.left_shoulder_camera.mask:
                obs[i].left_shoulder_mask = join(l_sh_mask_f, si)
            if obs_config.right_shoulder_camera.rgb:
                obs[i].right_shoulder_rgb = join(r_sh_rgb_f, si)
            if obs_config.right_shoulder_camera.depth or obs_config.right_shoulder_camera.point_cloud:
                obs[i].right_shoulder_depth = join(r_sh_depth_f, si)
            if obs_config.right_shoulder_camera.mask:
                obs[i].right_shoulder_mask = join(r_sh_mask_f, si)
            if obs_config.overhead_camera.rgb:
                obs[i].overhead_rgb = join(oh_rgb_f, si)
            if obs_config.overhead_camera.depth or obs_config.overhead_camera.point_cloud:
                obs[i].overhead_depth = join(oh_depth_f, si)
            if obs_config.overhead_camera.mask:
                obs[i].overhead_mask = join(oh_mask_f, si)
            if obs_config.wrist_camera.rgb:
                obs[i].wrist_rgb = join(wrist_rgb_f, si)
            if obs_config.wrist_camera.depth or obs_config.wrist_camera.point_cloud:
                obs[i].wrist_depth = join(wrist_depth_f, si)
            if obs_config.wrist_camera.mask:
                obs[i].wrist_mask = join(wrist_mask_f, si)
            if obs_config.front_camera.rgb:
                obs[i].front_rgb = join(front_rgb_f, si)
            if obs_config.front_camera.depth or obs_config.front_camera.point_cloud:
                obs[i].front_depth = join(front_depth_f, si)
            if obs_config.front_camera.mask:
                obs[i].front_mask = join(front_mask_f, si)

            # Remove low dim info if necessary
            if not obs_config.joint_velocities:
                obs[i].joint_velocities = None
            if not obs_config.joint_positions:
                obs[i].joint_positions = None
            if not obs_config.joint_forces:
                obs[i].joint_forces = None
            if not obs_config.gripper_open:
                obs[i].gripper_open = None
            if not obs_config.gripper_pose:
                obs[i].gripper_pose = None
            if not obs_config.gripper_joint_positions:
                obs[i].gripper_joint_positions = None
            if not obs_config.gripper_touch_forces:
                obs[i].gripper_touch_forces = None
            if not obs_config.task_low_dim_state:
                obs[i].task_low_dim_state = None

        if not image_paths:
            for i in range(num_steps):
                if obs_config.left_shoulder_camera.rgb:
                    obs[i].left_shoulder_rgb = np.array(
                        _resize_if_needed(
                            Image.open(obs[i].left_shoulder_rgb),
                            obs_config.left_shoulder_camera.image_size))
                if obs_config.right_shoulder_camera.rgb:
                    obs[i].right_shoulder_rgb = np.array(
                        _resize_if_needed(Image.open(
                        obs[i].right_shoulder_rgb),
                            obs_config.right_shoulder_camera.image_size))
                if obs_config.overhead_camera.rgb:
                    obs[i].overhead_rgb = np.array(
                        _resize_if_needed(Image.open(
                        obs[i].overhead_rgb),
                            obs_config.overhead_camera.image_size))
                if obs_config.wrist_camera.rgb:
                    obs[i].wrist_rgb = np.array(
                        _resize_if_needed(
                            Image.open(obs[i].wrist_rgb),
                            obs_config.wrist_camera.image_size))
                if obs_config.front_camera.rgb:
                    obs[i].front_rgb = np.array(
                        _resize_if_needed(
                            Image.open(obs[i].front_rgb),
                            obs_config.front_camera.image_size))

                if obs_config.left_shoulder_camera.depth or obs_config.left_shoulder_camera.point_cloud:
                    l_sh_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].left_shoulder_depth),
                            obs_config.left_shoulder_camera.image_size),
                        DEPTH_SCALE)
                    near = obs[i].misc['left_shoulder_camera_near']
                    far = obs[i].misc['left_shoulder_camera_far']
                    l_sh_depth_m = near + l_sh_depth * (far - near)
                    if obs_config.left_shoulder_camera.depth:
                        d = l_sh_depth_m if obs_config.left_shoulder_camera.depth_in_meters else l_sh_depth
                        obs[i].left_shoulder_depth = obs_config.left_shoulder_camera.depth_noise.apply(d)
                    else:
                        obs[i].left_shoulder_depth = None

                if obs_config.right_shoulder_camera.depth or obs_config.right_shoulder_camera.point_cloud:
                    r_sh_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].right_shoulder_depth),
                            obs_config.right_shoulder_camera.image_size),
                        DEPTH_SCALE)
                    near = obs[i].misc['right_shoulder_camera_near']
                    far = obs[i].misc['right_shoulder_camera_far']
                    r_sh_depth_m = near + r_sh_depth * (far - near)
                    if obs_config.right_shoulder_camera.depth:
                        d = r_sh_depth_m if obs_config.right_shoulder_camera.depth_in_meters else r_sh_depth
                        obs[i].right_shoulder_depth = obs_config.right_shoulder_camera.depth_noise.apply(d)
                    else:
                        obs[i].right_shoulder_depth = None

                if obs_config.overhead_camera.depth or obs_config.overhead_camera.point_cloud:
                    oh_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].overhead_depth),
                            obs_config.overhead_camera.image_size),
                        DEPTH_SCALE)
                    near = obs[i].misc['overhead_camera_near']
                    far = obs[i].misc['overhead_camera_far']
                    oh_depth_m = near + oh_depth * (far - near)
                    if obs_config.overhead_camera.depth:
                        d = oh_depth_m if obs_config.overhead_camera.depth_in_meters else oh_depth
                        obs[i].overhead_depth = obs_config.overhead_camera.depth_noise.apply(d)
                    else:
                        obs[i].overhead_depth = None

                if obs_config.wrist_camera.depth or obs_config.wrist_camera.point_cloud:
                    wrist_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].wrist_depth),
                            obs_config.wrist_camera.image_size),
                        DEPTH_SCALE)
                    near = obs[i].misc['wrist_camera_near']
                    far = obs[i].misc['wrist_camera_far']
                    wrist_depth_m = near + wrist_depth * (far - near)
                    if obs_config.wrist_camera.depth:
                        d = wrist_depth_m if obs_config.wrist_camera.depth_in_meters else wrist_depth
                        obs[i].wrist_depth = obs_config.wrist_camera.depth_noise.apply(d)
                    else:
                        obs[i].wrist_depth = None

                if obs_config.front_camera.depth or obs_config.front_camera.point_cloud:
                    front_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].front_depth),
                            obs_config.front_camera.image_size),
                        DEPTH_SCALE)
                    near = obs[i].misc['front_camera_near']
                    far = obs[i].misc['front_camera_far']
                    front_depth_m = near + front_depth * (far - near)
                    if obs_config.front_camera.depth:
                        d = front_depth_m if obs_config.front_camera.depth_in_meters else front_depth
                        obs[i].front_depth = obs_config.front_camera.depth_noise.apply(d)
                    else:
                        obs[i].front_depth = None

                if obs_config.left_shoulder_camera.point_cloud:
                    obs[i].left_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        l_sh_depth_m,
                        obs[i].misc['left_shoulder_camera_extrinsics'],
                        obs[i].misc['left_shoulder_camera_intrinsics'])
                if obs_config.right_shoulder_camera.point_cloud:
                    obs[i].right_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        r_sh_depth_m,
                        obs[i].misc['right_shoulder_camera_extrinsics'],
                        obs[i].misc['right_shoulder_camera_intrinsics'])
                if obs_config.overhead_camera.point_cloud:
                    obs[i].overhead_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        oh_depth_m,
                        obs[i].misc['overhead_camera_extrinsics'],
                        obs[i].misc['overhead_camera_intrinsics'])
                if obs_config.wrist_camera.point_cloud:
                    obs[i].wrist_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        wrist_depth_m,
                        obs[i].misc['wrist_camera_extrinsics'],
                        obs[i].misc['wrist_camera_intrinsics'])
                if obs_config.front_camera.point_cloud:
                    obs[i].front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        front_depth_m,
                        obs[i].misc['front_camera_extrinsics'],
                        obs[i].misc['front_camera_intrinsics'])

                # Masks are stored as coded RGB images.
                # Here we transform them into 1 channel handles.
                if obs_config.left_shoulder_camera.mask:
                    obs[i].left_shoulder_mask = rgb_handles_to_mask(
                        np.array(_resize_if_needed(Image.open(
                            obs[i].left_shoulder_mask),
                            obs_config.left_shoulder_camera.image_size)))
                if obs_config.right_shoulder_camera.mask:
                    obs[i].right_shoulder_mask = rgb_handles_to_mask(
                        np.array(_resize_if_needed(Image.open(
                            obs[i].right_shoulder_mask),
                            obs_config.right_shoulder_camera.image_size)))
                if obs_config.overhead_camera.mask:
                    obs[i].overhead_mask = rgb_handles_to_mask(
                        np.array(_resize_if_needed(Image.open(
                            obs[i].overhead_mask),
                            obs_config.overhead_camera.image_size)))
                if obs_config.wrist_camera.mask:
                    obs[i].wrist_mask = rgb_handles_to_mask(np.array(
                        _resize_if_needed(Image.open(
                            obs[i].wrist_mask),
                            obs_config.wrist_camera.image_size)))
                if obs_config.front_camera.mask:
                    try:
                        obs[i].front_mask = rgb_handles_to_mask(np.array(
                            _resize_if_needed(Image.open(
                                obs[i].front_mask),
                                obs_config.front_camera.image_size)))
                    except:
                        obs[i].front_mask = None

        demos.append(obs)
    return demos


def _resize_if_needed(image, size):
    if image.size[0] != size[0] or image.size[1] != size[1]:
        image = image.resize(size)
    return image
