import numpy as np
from scipy.linalg import logm, expm


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
    return tracker, updated



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


def chamfer_distance_birectional(set1, set2):
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
        chamfer_dist += np.mean(np.min(pairwise_distances, axis=0))

        return chamfer_dist * 0.5
