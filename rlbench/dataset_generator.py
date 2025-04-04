from absl import flags
import os
import pickle
import numpy as np
from PIL import Image
from rlbench.backend.const import *
from rlbench.backend import utils
# from .read_npy import process_demo, check_and_make
from .read_npy import check_and_make


FLAGS = flags.FLAGS

flags.DEFINE_string('save_path',
                    '/tmp/rlbench_data/',
                    'Where to save the demos.')
flags.DEFINE_list('tasks', [],
                  'The tasks to collect. If empty, all tasks are collected.')
flags.DEFINE_list('image_size', [128, 128],
                  'The size of the images tp save.')
flags.DEFINE_enum('renderer',  'opengl3', ['opengl', 'opengl3'],
                  'The renderer to use. opengl does not include shadows, '
                  'but is faster.')
flags.DEFINE_integer('processes', 1,
                     'The number of parallel processes during collection.')
flags.DEFINE_integer('episodes_per_task', 10,
                     'The number of episodes to collect per task.')
flags.DEFINE_integer('variations', -1,
                     'Number of variations to collect per task. -1 for all.')




def save_demo(scene_data, demo, example_path, variation):

    # Save image data first, and then None the image data, and pickle
    left_shoulder_rgb_path = os.path.join(
        example_path, LEFT_SHOULDER_RGB_FOLDER)
    left_shoulder_depth_path = os.path.join(
        example_path, LEFT_SHOULDER_DEPTH_FOLDER)
    left_shoulder_mask_path = os.path.join(
        example_path, LEFT_SHOULDER_MASK_FOLDER)
    right_shoulder_rgb_path = os.path.join(
        example_path, RIGHT_SHOULDER_RGB_FOLDER)
    right_shoulder_depth_path = os.path.join(
        example_path, RIGHT_SHOULDER_DEPTH_FOLDER)
    right_shoulder_mask_path = os.path.join(
        example_path, RIGHT_SHOULDER_MASK_FOLDER)
    overhead_rgb_path = os.path.join(
        example_path, OVERHEAD_RGB_FOLDER)
    overhead_depth_path = os.path.join(
        example_path, OVERHEAD_DEPTH_FOLDER)
    overhead_mask_path = os.path.join(
        example_path, OVERHEAD_MASK_FOLDER)
    wrist_rgb_path = os.path.join(example_path, WRIST_RGB_FOLDER)
    wrist_depth_path = os.path.join(example_path, WRIST_DEPTH_FOLDER)
    wrist_mask_path = os.path.join(example_path, WRIST_MASK_FOLDER)
    front_rgb_path = os.path.join(example_path, FRONT_RGB_FOLDER)
    front_depth_path = os.path.join(example_path, FRONT_DEPTH_FOLDER)
    front_mask_path = os.path.join(example_path, FRONT_MASK_FOLDER)

    left_shoulder_pcd_path = os.path.join(example_path, LEFT_SHOULDER_PCD_FOLDER)
    right_shoulder_pcd_path = os.path.join(example_path, RIGHT_SHOULDER_PCD_FOLDER)
    overhead_pcd_path = os.path.join(example_path, OVERHEAD_PCD_FOLDER)
    front_pcd_path = os.path.join(example_path, FRONT_PCD_FOLDER)

    check_and_make(left_shoulder_rgb_path)
    check_and_make(left_shoulder_depth_path)
    check_and_make(left_shoulder_mask_path)
    check_and_make(left_shoulder_pcd_path)
    check_and_make(right_shoulder_rgb_path)
    check_and_make(right_shoulder_depth_path)
    check_and_make(right_shoulder_mask_path)
    check_and_make(right_shoulder_pcd_path)
    check_and_make(overhead_rgb_path)
    check_and_make(overhead_depth_path)
    check_and_make(overhead_mask_path)
    check_and_make(overhead_pcd_path)
    check_and_make(wrist_rgb_path)
    check_and_make(wrist_depth_path)
    check_and_make(wrist_mask_path)
    check_and_make(front_rgb_path)
    check_and_make(front_depth_path)
    check_and_make(front_mask_path)
    check_and_make(front_pcd_path)


    tot_contact_info = {}
    tot_pose_info = {}
    for i, obs in enumerate(demo):
        left_shoulder_rgb = Image.fromarray(obs.left_shoulder_rgb)
        left_shoulder_depth = utils.float_array_to_rgb_image(
            obs.left_shoulder_depth, scale_factor=DEPTH_SCALE)
        left_shoulder_mask = Image.fromarray(
            (obs.left_shoulder_mask * 255).astype(np.uint8))
        right_shoulder_rgb = Image.fromarray(obs.right_shoulder_rgb)
        right_shoulder_depth = utils.float_array_to_rgb_image(
            obs.right_shoulder_depth, scale_factor=DEPTH_SCALE)
        right_shoulder_mask = Image.fromarray(
            (obs.right_shoulder_mask * 255).astype(np.uint8))
        overhead_rgb = Image.fromarray(obs.overhead_rgb)
        overhead_depth = utils.float_array_to_rgb_image(
            obs.overhead_depth, scale_factor=DEPTH_SCALE)
        overhead_mask = Image.fromarray(
            (obs.overhead_mask * 255).astype(np.uint8))
        wrist_rgb = Image.fromarray(obs.wrist_rgb)
        wrist_depth = utils.float_array_to_rgb_image(
            obs.wrist_depth, scale_factor=DEPTH_SCALE)
        wrist_mask = Image.fromarray((obs.wrist_mask * 255).astype(np.uint8))
        front_rgb = Image.fromarray(obs.front_rgb)
        front_depth = utils.float_array_to_rgb_image(
            obs.front_depth, scale_factor=DEPTH_SCALE)
        front_mask = Image.fromarray((obs.front_mask * 255).astype(np.uint8))

        # Save poincloud to pkl
        with open(os.path.join(left_shoulder_pcd_path, PCD_FORMAT % i), 'wb') as f:
            pickle.dump( np.float16(obs.left_shoulder_point_cloud), f)
        with open(os.path.join(right_shoulder_pcd_path, PCD_FORMAT % i), 'wb') as f:
            pickle.dump(np.float16(obs.right_shoulder_point_cloud), f) 
        with open(os.path.join(overhead_pcd_path, PCD_FORMAT % i), 'wb') as f:
            pickle.dump(np.float16(obs.overhead_point_cloud), f)             
        with open(os.path.join(front_pcd_path, PCD_FORMAT % i), 'wb') as f:
            pickle.dump(np.float16(obs.front_point_cloud), f)

        left_shoulder_rgb.save(
            os.path.join(left_shoulder_rgb_path, IMAGE_FORMAT % i))
        left_shoulder_depth.save(
            os.path.join(left_shoulder_depth_path, IMAGE_FORMAT % i))
        left_shoulder_mask.save(
            os.path.join(left_shoulder_mask_path, IMAGE_FORMAT % i))
        right_shoulder_rgb.save(
            os.path.join(right_shoulder_rgb_path, IMAGE_FORMAT % i))
        right_shoulder_depth.save(
            os.path.join(right_shoulder_depth_path, IMAGE_FORMAT % i))
        right_shoulder_mask.save(
            os.path.join(right_shoulder_mask_path, IMAGE_FORMAT % i))
        overhead_rgb.save(
            os.path.join(overhead_rgb_path, IMAGE_FORMAT % i))
        overhead_depth.save(
            os.path.join(overhead_depth_path, IMAGE_FORMAT % i))
        overhead_mask.save(
            os.path.join(overhead_mask_path, IMAGE_FORMAT % i))
        wrist_rgb.save(os.path.join(wrist_rgb_path, IMAGE_FORMAT % i))
        wrist_depth.save(os.path.join(wrist_depth_path, IMAGE_FORMAT % i))
        wrist_mask.save(os.path.join(wrist_mask_path, IMAGE_FORMAT % i))
        front_rgb.save(os.path.join(front_rgb_path, IMAGE_FORMAT % i))
        front_depth.save(os.path.join(front_depth_path, IMAGE_FORMAT % i))
        front_mask.save(os.path.join(front_mask_path, IMAGE_FORMAT % i))

        # We save the images separately, so set these to None for pickling.
        obs.left_shoulder_rgb = None
        obs.left_shoulder_depth = None
        obs.left_shoulder_point_cloud = None
        obs.left_shoulder_mask = None
        obs.right_shoulder_rgb = None
        obs.right_shoulder_depth = None
        obs.right_shoulder_point_cloud = None
        obs.right_shoulder_mask = None
        obs.overhead_rgb = None
        obs.overhead_depth = None
        obs.overhead_point_cloud = None
        obs.overhead_mask = None
        obs.wrist_rgb = None
        obs.wrist_depth = None
        obs.wrist_point_cloud = None
        obs.wrist_mask = None
        obs.front_rgb = None
        obs.front_depth = None
        obs.front_point_cloud = None
        obs.front_mask = None

        tot_contact_info[i] = obs.contact_info
        tot_pose_info[i] = obs.object_poses

    # Save camera params & extrinsics
    front_intrinsics = [scene_data['front_camera']['intrinsics'][0,0], 
                scene_data['front_camera']['intrinsics'][1,1], 
                scene_data['front_camera']['intrinsics'][0,2], 
                scene_data['front_camera']['intrinsics'][1,2]]


    left_intrinsics = [scene_data['left_shoulder_camera']['intrinsics'][0,0], 
                       scene_data['left_shoulder_camera']['intrinsics'][1,1], 
                       scene_data['left_shoulder_camera']['intrinsics'][0,2], 
                       scene_data['left_shoulder_camera']['intrinsics'][1,2]]

    right_intrinsics = [scene_data['right_shoulder_camera']['intrinsics'][0,0], 
                       scene_data['right_shoulder_camera']['intrinsics'][1,1], 
                       scene_data['right_shoulder_camera']['intrinsics'][0,2], 
                       scene_data['right_shoulder_camera']['intrinsics'][1,2]]

    front_extrinsics = scene_data['front_camera']['extrinsics']
    left_extrinsics = scene_data['left_shoulder_camera']['extrinsics']
    right_extrinsics = scene_data['right_shoulder_camera']['extrinsics']


    

    # Save contact info
    with open(os.path.join(example_path, "contact_info.pkl"), 'wb') as f:
        pickle.dump(tot_contact_info, f)

    # # Save contact info
    # for rigid transform for finding source contacts
    with open(os.path.join(example_path, "object_poses.pkl"), 'wb') as f:
        pickle.dump(tot_pose_info, f)

    np.save(os.path.join(example_path, "right_shoulder_camera_extrinsics.npy"), right_extrinsics)
    np.save(os.path.join(example_path, "right_shoulder_camera_params.npy"), right_intrinsics)

    np.save(os.path.join(example_path, "left_shoulder_camera_extrinsics.npy"), left_extrinsics)
    np.save(os.path.join(example_path, "left_shoulder_camera_params.npy"), left_intrinsics)

    np.save(os.path.join(example_path, "front_camera_extrinsics.npy"), front_extrinsics)
    np.save(os.path.join(example_path, "front_camera_params.npy"), front_intrinsics)



    # Save the low-dimension data
    with open(os.path.join(example_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(demo, f)

    with open(os.path.join(example_path, VARIATION_NUMBER), 'wb') as f:
        pickle.dump(variation, f)
