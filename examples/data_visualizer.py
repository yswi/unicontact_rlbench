import numpy as np
import os
import copy
import pickle
import open3d as o3d
from random import uniform as u
from vedo import settings, Points, Arrows, Plotter
from PIL import Image
from scipy.linalg import logm, expm
from rlbench.backend.const import *
from scipy.spatial.transform import Rotation as R

# task specific param.
task_name = 'sweep_to_dustpan'
variation_num = 0
vis = True

# pathes.
SAVE_PATH = os.path.abspath( os.path.join(os.path.dirname(__file__), '../../dataset/'))
variation_path = os.path.join(SAVE_PATH, task_name, VARIATIONS_ALL_FOLDER)
episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
example_path = os.path.join(episodes_path, EPISODE_FOLDER % variation_num)

settings.default_font = "Calco"
contact_point_color = ["blue5", "orange5", "orange5", "pink5", "indigo5"]
relative_pose_tracker = {}

def update_relative_pose_tracker(obj1, obj2, pose, tracker, eps):
    updated = False
    if obj1 in tracker.keys():
        if obj2 in tracker[obj1].keys():
            # Update if meaningful changes.
            if np.linalg.norm( np.array(pose)[:3] - np.array(tracker[obj1][obj2])[:3]) > eps:
                tracker[obj1][obj2]= pose
                updated = True
        else:
            tracker[obj1][obj2]= pose
            updated = True
    else:
        tracker[obj1] = {obj2: pose}
        updated = True


def get_rotation_matrix(v1, v2):
    """Get the rotation matrix that aligns vectors v1 and v2."""
    
    # normalize vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # cross product to get an axis of rotation
    axis = np.cross(v1, v2)
    axis = axis / np.linalg.norm(axis)
    
    # angle between two vectors
    theta = np.arccos(np.dot(v1, v2))
    
    # Rodrigues' rotation formula
    R = expm(-theta * np.cross(np.eye(3), axis))
    
    return R

def chamfer_distance_directional(set1, set2):
        """
        Compute Chamfer distance between two sets of points.
        
        Args:
        - set1 (torch.Tensor): Set of points, shape (N, D).
        - set2 (torch.Tensor): Set of points, shape (M, D).
        
        Returns:
        - chamfer_dist (torch.Tensor): Chamfer distance between the sets.
        """
        
        def pairwise_distance_matrix(set1, set2):
            """
            Compute pairwise distances between points in two sets.
            
            Args:
            - set1 (torch.Tensor): Set of points, shape (N, D).
            - set2 (torch.Tensor): Set of points, shape (M, D).
            
            Returns:
            - distances (torch.Tensor): Pairwise distance matrix, shape (N, M).
            """
            # Compute squared distances
            set1_norm = np.sum(set1**2, axis=1, keepdims=True)
            set2_norm = np.sum(set2**2, axis=1, keepdims=True)
            distances = set1_norm + set2_norm.transpose() - 2 * np.matmul(set1, set2.transpose())
            return distances
        
        pairwise_distances = pairwise_distance_matrix(set1, set2)
        chamfer_dist = np.mean(np.min(pairwise_distances, axis=1))
        return chamfer_dist


if __name__ == '__main__':
    contact_dict = np.load(os.path.join(example_path, "contact_info.pkl"), allow_pickle=True)
    object_poses_dict = np.load(os.path.join(example_path, "object_poses.pkl"), allow_pickle=True)
    assert len(contact_dict) == len(object_poses_dict)

    with open(os.path.join(example_path, LOW_DIM_PICKLE), 'rb') as f:
        load_dim_data = pickle.load(f)

    vis_o3d = o3d.visualization.Visualizer()
    vis_o3d.create_window()
    grasped = False
    contact_pcd_prev = []
    contact_pcd = []
    time_stamps = list(contact_dict.keys())
    for idx_t in range(len(time_stamps)):
        time_stamp = time_stamps[idx_t]

        contact_dict_i = contact_dict[time_stamp]
        object_poses_i = object_poses_dict[time_stamp]
        contact_dict_prev = contact_dict[idx_t-1] if idx_t !=0 else None
        object_poses_prev = object_poses_dict[idx_t-1]  if idx_t !=0 else None
        load_dim_data_i = load_dim_data[idx_t]


        if vis and load_dim_data_i.keyframe:

            # visualize 3d contact
            plt = Plotter(N=1, axes=1)
            pcd = np.load(os.path.join(example_path, f"{FRONT_PCD_FOLDER}/{time_stamp}.pkl"), allow_pickle=True).reshape(-1,3)
            rgb = np.array(Image.open(os.path.join(example_path, f"{FRONT_RGB_FOLDER}/{time_stamp}.png")))

            pcd_ls = np.load(os.path.join(example_path, f"{LEFT_SHOULDER_PCD_FOLDER}/{time_stamp}.pkl"), allow_pickle=True).reshape(-1,3)
            rgb_ls = np.array(Image.open(os.path.join(example_path, f"{LEFT_SHOULDER_RGB_FOLDER}/{time_stamp}.png")))

            pcd_rs = np.load(os.path.join(example_path, f"{RIGHT_SHOULDER_PCD_FOLDER}/{time_stamp}.pkl"), allow_pickle=True).reshape(-1,3)
            rgb_rs = np.array(Image.open(os.path.join(example_path, f"{RIGHT_SHOULDER_RGB_FOLDER}/{time_stamp}.png")))

            # Create a new PointCloud object and add points and colors to it
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
            pcd_o3d.colors = o3d.utility.Vector3dVector(rgb.reshape(-1,3)/255)

            # Create a new PointCloud object and add points and colors to it
            pcd_ls_o3d = o3d.geometry.PointCloud()
            pcd_ls_o3d.points = o3d.utility.Vector3dVector(pcd_ls)
            pcd_ls_o3d.colors = o3d.utility.Vector3dVector(rgb_ls.reshape(-1,3)/255)

            # Create a new PointCloud object and add points and colors to it
            pcd_rs_o3d = o3d.geometry.PointCloud()
            pcd_rs_o3d.points = o3d.utility.Vector3dVector(pcd_rs)
            pcd_rs_o3d.colors = o3d.utility.Vector3dVector(rgb_rs.reshape(-1,3)/255)
            

        contacts = []
        for source_idx, source_contact_cand in enumerate(contact_dict_i.keys()) :
            pts = contact_dict_i[source_contact_cand]['points']
            forces = contact_dict_i[source_contact_cand]['forces']
            cnt_handles = contact_dict_i[source_contact_cand]['cnt_handles']

            # Get relative pose
            new_contact_idx = []
            relative_pose = []
            for cnt_handle_i in cnt_handles:
                cnt_handle_i = cnt_handle_i.astype(int)
                pose = contact_dict_i[source_contact_cand]['relative_pose'][cnt_handle_i[0]][cnt_handle_i[1]] # relative pose of contacts

                # delta object pose per each contact
                if time_stamp != 0:
                    # Current tool pose - Previous tool pose in world 
                    relative_pose_i = np.array(object_poses_i[cnt_handle_i[0]][:3]) - np.array(object_poses_prev[cnt_handle_i[0]][:3])
                    relative_pose.append(relative_pose_i)
                else: 
                    relative_pose.append(np.array([0, 0, 0, 0 ,0 ,0, 0]))

            if vis and idx_t >0 and len(pts) > 0 and load_dim_data_i.keyframe:
                
                color = np.array([0., 0., 0.])
                color[source_idx] = 1.
                
                for i, pt_i in enumerate(pts):
                    f_i = forces[i]
                    cnt_handle_i = cnt_handles[i].astype(int)
                    if np.linalg.norm(f_i) > 0:

                        ### Filter 1: Remove contact haven't move in the world frame 
                        ### Filter 1.1 : Directly remove that was at the previous timeframe
                        if len(contact_dict_prev[source_contact_cand]['points']) > 0: # Given valid prev contact
                            if pt_i in contact_dict_prev[source_contact_cand]['points']: # check if there's static contact
                                continue
                        
                        if grasped:
                            ### Approach angle @ grasping = delta tool pose
                            approach_angle = relative_pose[i][:3]

                            # ### Filter 2: Remove contact near gripper after grasping (when gripper force > 0.1 N)
                            # if np.linalg.norm(np.array(load_dim_data_i.gripper_pose[:3]) - pt_i[:3]) < 0.03:
                            #     continue

                        else:
                            ### Approach angle @ grasping = gripper pose                            
                            r = R.from_quat(load_dim_data_i.gripper_pose[3:7])
                            approach_angle = r.apply([0,0,1])

                            ### Filter 3: Focus on grasping (when gripper force <= 0.1 N)
                            if not np.linalg.norm(np.array(load_dim_data_i.gripper_pose[:3]) - pt_i[:3]) < 0.03:
                                continue

                            ### Filter 4: Contact has to be mutual. (we only do that during grasping)
                            if cnt_handle_i[1] not in list(contact_dict_i[source_contact_cand]['relative_pose'].keys()):
                                continue
                            if cnt_handle_i[0] not in list(contact_dict_i[source_contact_cand]['relative_pose'][cnt_handle_i[1]]):
                                continue

                        # Visualize approach angle
                        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.005, cone_radius=0.005, 
                                                    cylinder_height= 0.1, cone_height=0.05)
                        arrow.paint_uniform_color(1 - color)  
                        T = get_rotation_matrix(np.array(approach_angle), np.array([0,0,1]))
                        arrow.rotate(T, center=(0, 0, 0))  # Apply rotation
                        arrow.translate((pt_i[0], pt_i[1], pt_i[2]))   # Translation vector (x, y, z)
                        contacts.append(arrow)

                        # Visualize contact location
                        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                        sphere.paint_uniform_color(color)  # RGB value for green color
                        sphere.translate((pt_i[0], pt_i[1], pt_i[2]))   # Translation vector (x, y, z)
                        contacts.append(sphere) # reaction force
                       
                        # Visualize contact force
                        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.005, cone_radius=0.005, 
                                                    cylinder_height= np.clip(np.linalg.norm(f_i)* 0.05, 0, 0.1), cone_height=0.05)
                        arrow.paint_uniform_color(color)  
                        T = get_rotation_matrix( -np.array(f_i),  np.array([0,0,1]))
                        arrow.rotate(T, center=(0, 0, 0))  # Apply rotation
                        arrow.translate((pt_i[0], pt_i[1], pt_i[2]))   # Translation vector (x, y, z)
                        contacts.append(arrow)

                        if grasped:
                            # make environment contact force acting on the env
                            arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.005, cone_radius=0.005, 
                                                        cylinder_height= np.clip(np.linalg.norm(f_i)* 0.05, 0, 0.1), cone_height=0.05)
                            arrow.paint_uniform_color([0, 1, 0])  
                            T = get_rotation_matrix( np.array(f_i),  np.array([0,0,1]))
                            arrow.rotate(T, center=(0, 0, 0))  # Apply rotation
                            arrow.translate((pt_i[0], pt_i[1], pt_i[2]))   # Translation vector (x, y, z)
                            contacts.append(arrow)

                        contact_pcd.append([pt_i[0], pt_i[1], pt_i[2]])
        
        
        ### Filter 1.2: Get the chamfer-distance between the previous keyframe's contacts and the current. Remove when it's the same.
        cd = 0.1
        if len(contact_pcd_prev) > 0:
            cd = chamfer_distance_directional(np.stack(contact_pcd, axis = 0), np.stack(contact_pcd_prev, axis = 0) )
            print("chamfer distance", cd)

        # After successful grasp        
        grasped = True if np.linalg.norm(load_dim_data_i.gripper_touch_forces) > 0.1 else False
        if vis and load_dim_data_i.keyframe and len(contacts)>0 and cd > 1e-4:
            pcd_o3d = pcd_o3d.voxel_down_sample(voxel_size=0.02)
            pcd_ls_o3d = pcd_ls_o3d.voxel_down_sample(voxel_size=0.02)
            pcd_rs_o3d = pcd_rs_o3d.voxel_down_sample(voxel_size=0.02)

            vis_o3d.add_geometry(pcd_o3d)
            vis_o3d.add_geometry(pcd_ls_o3d)
            vis_o3d.add_geometry(pcd_rs_o3d)
            for contact_i in contacts:
                vis_o3d.add_geometry(contact_i)

            vis_o3d.reset_view_point()
            cntrl = vis_o3d.get_view_control()

            cntrl.set_up([0, 0, 1])
            cntrl.set_front([0.9, 0.45, 0.4])
            cntrl.set_zoom(0.2)

            prev_contact = copy.copy(contact_dict_i)
            vis_o3d.update_renderer()

            vis_o3d.capture_screen_image(f"{example_path}/keyframe_vis/contact_vis_{time_stamp}.png", do_render=True)

            vis_o3d.remove_geometry(pcd_o3d, reset_bounding_box=False)
            vis_o3d.remove_geometry(pcd_ls_o3d, reset_bounding_box=False)
            vis_o3d.remove_geometry(pcd_rs_o3d, reset_bounding_box=False)
            for contact_i in contacts:
                vis_o3d.remove_geometry(contact_i, reset_bounding_box=False)

            contact_pcd_prev = copy.copy(contact_pcd)