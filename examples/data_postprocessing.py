import numpy as np
import os
import copy
import pickle
import open3d as o3d
from random import uniform as u
# from vedo import settings, Points, Arrows, Plotter
from PIL import Image
from rlbench.backend.const import *
from scipy.spatial.transform import Rotation as R
from rlbench.demo_utils import *
from rlbench.read_npy import check_and_make
import shutil

# task specific param.
task_name = 'sweep_to_dustpan'
variation_num = 0

## workspace bounding box (for visualization purpose only)
x_upper = 1.
x_lower = -1.
y_upper = .8
y_lower = -.8
z_upper = 1.8
z_lower = 0.

# pathes.
SAVE_PATH = os.path.abspath( os.path.join(os.path.dirname(__file__), '../../dataset/'))
variation_path = os.path.join(SAVE_PATH, task_name, VARIATIONS_ALL_FOLDER)
episodes_path = os.path.join(variation_path, EPISODES_FOLDER)

# settings.default_font = "Calco"
contact_point_color = ["blue5", "orange5", "orange5", "pink5", "indigo5"]
relative_pose_tracker = {}



if __name__ == '__main__':
    vis_o3d = o3d.visualization.Visualizer()
    vis_o3d.create_window()
    grasped = False
    for variation_num in range(100):
        example_path = os.path.join(episodes_path, EPISODE_FOLDER % variation_num)
        contact_dict = np.load(os.path.join(example_path, "contact_info.pkl"), allow_pickle=True)
        object_poses_dict = np.load(os.path.join(example_path, "object_poses.pkl"), allow_pickle=True)
        keyframe_path = os.path.join(example_path, "keyframe_summary")
        if os.path.isdir(keyframe_path):
            shutil.rmtree(keyframe_path)
        check_and_make(keyframe_path) 
        assert len(contact_dict) == len(object_poses_dict)

        with open(os.path.join(example_path, LOW_DIM_PICKLE), 'rb') as f:
            load_dim_data = pickle.load(f)



        time_stamps = list(contact_dict.keys())

        keyframe_infos = {"timestamps": [0], # keyframe always includes reset 
                        "source_contact": [],
                        "target_contact": [],
                        "approach_angle": [],
                        "source_pose": [object_poses_dict[0]],
                        "contact_force": []} 
        contact_pcd_prev = []

        for idx_t in range(1, len(time_stamps)):
            time_stamp = time_stamps[idx_t]
            time_stamp_prev = time_stamps[idx_t-1]
            contact_pcd = []
            prev_contact = contact_dict[time_stamp_prev] 
            contact_dict_i = contact_dict[time_stamp]
            object_poses_i = object_poses_dict[time_stamp]
            contact_dict_prev = contact_dict[idx_t-1] if idx_t !=0 else None
            object_poses_prev = object_poses_dict[idx_t-1]  if idx_t !=0 else None
            load_dim_data_i = load_dim_data[idx_t]
            load_dim_data_prev = load_dim_data[idx_t-1]
            

            contacts = []
            tot_contact_force = np.array([0., 0., 0.])
            approach_angle = np.array([0., 0., 0.])

            if load_dim_data_i.keyframe:
                # visualize 3d contact
                # plt = Plotter(N=1, axes=1)
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
                

            for source_idx, source_contact_cand in enumerate(contact_dict_i.keys()) :
                pts = contact_dict_i[source_contact_cand]['points']
                forces = contact_dict_i[source_contact_cand]['forces']
                normals = contact_dict_i[source_contact_cand]['normals']
                cnt_handles = contact_dict_i[source_contact_cand]['cnt_handles']

                # Get relative pose
                new_contact_idx = []
                relative_pose = []
                for cnt_handle_i in cnt_handles:
                    cnt_handle_i = cnt_handle_i.astype(int)
                    pose = contact_dict_i[source_contact_cand]['relative_pose'][cnt_handle_i[0]][cnt_handle_i[1]] # relative pose of contacts

                    # delta object pose per each contact
                    relative_pose_i = np.array([0, 0, 0, 0 ,0 ,0, 0])
                    if time_stamp != 0:
                        # Current tool pose - Previous tool pose in world 
                        relative_pose_i = np.array(object_poses_i[cnt_handle_i[0]][:3]) - keyframe_infos["source_pose"][-1][cnt_handle_i[0]][:3] #- np.array(object_poses_prev[cnt_handle_i[0]][:3])
                    relative_pose.append(relative_pose_i)


                if idx_t >0 and len(pts) > 0 and load_dim_data_i.keyframe:
                    
                    color = np.array([0., 0., 0.])
                    color[source_idx] = 1.
                    contact_indexes = []

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

                                ### Filter 2.1: Remove noisy contacts near grippers -todo: not sure why this does not solve 2.2
                                if np.linalg.norm(np.array(load_dim_data_i.gripper_pose[:3]) - pt_i[:3]) < 0.06:
                                    continue

                            else:
                                ### Approach angle @ grasping = gripper pose                            
                                r = R.from_quat(load_dim_data_i.gripper_pose[3:7])
                                approach_angle = r.apply([0,0,1])

                                ### Filter 4.2: graping only happens on finger
                                if cnt_handle_i[0] != contact_dict_i[source_contact_cand]["labels"]["Panda_leftfinger_force_contact"]:
                                    continue


                            # Visualize approach angle
                            arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.005, cone_radius=0.005, 
                                                        cylinder_height= 0.1, cone_height=0.01)
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
                                                        cylinder_height= np.clip(np.linalg.norm(f_i)* 0.01, 0, 0.1), cone_height=0.05)
                            arrow.paint_uniform_color(color)  
                            T = get_rotation_matrix( -np.array(f_i),  np.array([0,0,1]))
                            arrow.rotate(T, center=(0, 0, 0))  # Apply rotation
                            arrow.translate((pt_i[0], pt_i[1], pt_i[2]))   # Translation vector (x, y, z)
                            contacts.append(arrow)

                            if grasped:
                                # make environment contact force acting on the env
                                arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.005, cone_radius=0.005, 
                                                            cylinder_height= np.clip(np.linalg.norm(f_i)* 0.01, 0, 0.1), cone_height=0.05)
                                arrow.paint_uniform_color([0, 1, 0])  
                                T = get_rotation_matrix( np.array(f_i),  np.array([0,0,1]))
                                arrow.rotate(T, center=(0, 0, 0))  # Apply rotation
                                arrow.translate((pt_i[0], pt_i[1], pt_i[2]))   # Translation vector (x, y, z)
                                contacts.append(arrow)

                            contact_pcd.append([pt_i[0], pt_i[1], pt_i[2]])
                            tot_contact_force -= f_i
                            contact_indexes.append(i)

                    
            
            ### Filter 1.2: Get the chamfer-distance between the previous keyframe's contacts and the current. Remove when it's the same.
            cd = 0.1
            if len(contact_pcd_prev) > 2 and len(contact_pcd) > 0:
                cd = chamfer_distance_directional(np.stack(contact_pcd, axis = 0), np.stack(contact_pcd_prev, axis = 0) )
            
            # Save keyframes     
            grasped = True if np.linalg.norm(load_dim_data_i.gripper_touch_forces) > 0.01 else False
            if load_dim_data_i.keyframe and len(contact_pcd)>2 and cd > 1e-4 and np.linalg.norm(approach_angle) > 1e-2:
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

                vis_o3d.update_renderer()
                vis_o3d.capture_screen_image(os.path.join(keyframe_path, f"contact_vis_{time_stamp}.png") , do_render=True)
                keyframe_infos["timestamps"].append(time_stamp)
                keyframe_infos["approach_angle"].append(approach_angle)
                keyframe_infos["source_pose"].append(object_poses_i) 
                keyframe_infos["target_contact"].append(np.stack(contact_pcd, axis = 0))
                keyframe_infos["contact_force"].append(tot_contact_force)

                # Source contact: rigid transform target contact 
                ## Get the delta transform of source object between keypoints
                source_pose_cur = keyframe_infos["source_pose"][-1][cnt_handle_i[0]]
                source_pose_prev = keyframe_infos["source_pose"][-2][cnt_handle_i[0]]
                new_contact = np.stack(contact_pcd, axis = 0)

                # delta: prev -> cur
                delta_orientation = R.from_quat(source_pose_cur[3:7]).as_matrix() @ R.from_quat(source_pose_prev[3:7]).inv().as_matrix()
                new_contact[...,:3] -= source_pose_cur[:3]
                
                # rotate 
                # r_action = R.from_euler('xyz', action[3:6])
                # new_contact = delta_orientation.inv().apply(new_contact)
                new_contact = R.from_matrix(delta_orientation).inv().apply(new_contact)
                new_contact[...,:3] += source_pose_cur[:3]
                
                # TODO: build a transform matrix and apply to the point
                delta_position = np.array(source_pose_cur[:3]) - np.array(source_pose_prev[:3])
                keyframe_infos["source_contact"].append(new_contact - delta_position)

                vis_o3d.remove_geometry(pcd_o3d, reset_bounding_box=False)
                vis_o3d.remove_geometry(pcd_ls_o3d, reset_bounding_box=False)
                vis_o3d.remove_geometry(pcd_rs_o3d, reset_bounding_box=False)
                for contact_i in contacts:
                    vis_o3d.remove_geometry(contact_i, reset_bounding_box=False)

                contact_pcd_prev = copy.copy(contact_pcd)


        # Save post-processed dataset
        with open(os.path.join(example_path, "dataset.pkl"), 'wb') as f:
            pickle.dump(keyframe_infos, f)



        # Visualize the keyframe and save it.
        for i in range( len(keyframe_infos["timestamps"][1:])):
            timestamp = keyframe_infos["timestamps"][i]
            # plt = Plotter(N=1, axes=1)
            pcd = np.load(os.path.join(example_path, f"{FRONT_PCD_FOLDER}/{timestamp}.pkl"), allow_pickle=True).reshape(-1,3)
            rgb = np.array(Image.open(os.path.join(example_path, f"{FRONT_RGB_FOLDER}/{timestamp}.png")))
                
            # Create a new PointCloud object and add points and colors to it
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_idx = np.where((pcd[:,0] > x_lower ) & (pcd[:,0] < x_upper ))[0]
            pcd_o3d.points = o3d.utility.Vector3dVector(pcd[pcd_idx])
            pcd_o3d.colors = o3d.utility.Vector3dVector(rgb.reshape(-1,3)[pcd_idx]/255)


            c_target = keyframe_infos["target_contact"][i]
            c_source = keyframe_infos["source_contact"][i]
            c_force = keyframe_infos["contact_force"][i]
            approach_angle = keyframe_infos["approach_angle"][i]

            # Visualize approach angle @ target
            keypoint_contact = []
            for pt_i in c_target:
                arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.005, cone_radius=0.005, 
                                            cylinder_height=np.linalg.norm(approach_angle)*0.15, cone_height=0.01)
                arrow.paint_uniform_color([1, 0, 0])  
                T = get_rotation_matrix(np.array(approach_angle), np.array([0,0,1]))
                arrow.rotate(T, center=(0, 0, 0))  # Apply rotation
                arrow.translate((pt_i[0], pt_i[1], pt_i[2]))   # Translation vector (x, y, z)
                keypoint_contact.append(arrow)

            # Visualize target contact
            for pt_i in c_target:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
                sphere.paint_uniform_color([1, 0.35, 0])  # RGB value for green color
                sphere.translate((pt_i[0], pt_i[1], pt_i[2]))   # Translation vector (x, y, z)
                keypoint_contact.append(sphere) # reaction force

            # Visualize source contact
            for pt_i in c_source:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
                sphere.paint_uniform_color([0,0,1])  # RGB value for green color
                sphere.translate((pt_i[0], pt_i[1], pt_i[2]))   # Translation vector (x, y, z)
                keypoint_contact.append(sphere) # reaction force
                
            # Visualize contact force
            for pt_i in c_target:
                arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.005, cone_radius=0.005, 
                                            cylinder_height= np.clip(np.linalg.norm(c_force)* 0.0001, 0.0001, 0.1), cone_height=0.01)
                arrow.paint_uniform_color([ 0, 1, 0])  
                T = get_rotation_matrix( -np.array(c_force),  np.array([0,0,1]))
                arrow.rotate(T, center=(0, 0, 0))  # Apply rotation
                arrow.translate((pt_i[0], pt_i[1], pt_i[2]))   # Translation vector (x, y, z)
                keypoint_contact.append(arrow)


            pcd_o3d = pcd_o3d.voxel_down_sample(voxel_size=0.02)
            vis_o3d.add_geometry(pcd_o3d)

            for contact_i in keypoint_contact:
                vis_o3d.add_geometry(contact_i)

            vis_o3d.reset_view_point()
            cntrl = vis_o3d.get_view_control()

            cntrl.set_up([ -0.3183616251567537, -0.17305138314817262, 0.93204028583428467])
            cntrl.set_front([0.71723555337395828, 0.59891705380412286, 0.35619029133167218 ])
            cntrl.set_zoom(0.23)

            prev_contact = copy.copy(contact_dict_i)
            vis_o3d.update_renderer()
            vis_o3d.capture_screen_image(os.path.join(keyframe_path, f"contact_dataset_{timestamp}.png"), do_render=True)

            vis_o3d.remove_geometry(pcd_o3d, reset_bounding_box=False)
            for contact_i in keypoint_contact:
                vis_o3d.remove_geometry(contact_i, reset_bounding_box=False)
