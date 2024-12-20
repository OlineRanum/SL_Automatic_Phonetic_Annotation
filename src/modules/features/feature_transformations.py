import numpy as np 


class BaseFeatures:
    def __init__(self, pose_data):
        print("BaseFeatures initialized.")
        self.keypoints_body = pose_data['keypoints_body']
        self.keypoints_hand_left = pose_data['keypoints_hand_left']
        self.keypoints_hand_right = pose_data['keypoints_hand_right']
        self.normalized_keypoints_hand_left = pose_data['normalized_keypoints_hand_left']
        self.normalized_keypoints_hand_right = pose_data['normalized_keypoints_hand_right']


    
class MaskFeatures(BaseFeatures):
    def __init__(self, mask_type):
        
        if mask_type == 'wrist_to_fingers':
            self.mask = self.wrist_to_finger_mask()
        elif mask_type == 'gomer':
            indexes = [[0,1], [0,4], [0,5], [0,8], [0,9], [0,12], [0,13], [0,16], [0,17], [0,20], [3,2], [3,4], [6,7], [6,8], [10,11], [10,12], [14,15], [14,16], [18,19], [18,20]]
            self.mask = self.index_mask(indexes)

    def mask_features(self, pdm):
        """
        Input: pdm (numpy.ndarray): A 21x21 pairwise distance matrix.
        Output: numpy.ndarray: A masked pairwise distance matrix.
        """ 
        return pdm * self.mask
    
    def wrist_to_finger_mask(self):
        mask = np.zeros((21, 21))
        mask[0, 1:] = 1
        mask[1:, 0] = 1
        return mask
    
    def index_mask(self, indexes):
        mask = np.zeros((21, 21))
        for index in indexes:
            mask[index[0], index[1]] = 1
            mask[index[1], index[0]] = 1
        return mask

class DistanceFeatures(BaseFeatures):
    def __init__(self):
        pass
        #self.dist_matrix = self.pairwise_distance_matrix(points)
        
    def pairwise_distance_matrix(self, points):
        """
        Calculate the normalized symmetric pairwise distance matrix for a given 3D point cloud.
        
        Parameters:
            points (numpy.ndarray): An array of shape (21, 3) representing the 3D coordinates of the hand nodes.
            
        Returns:
            numpy.ndarray: A normalized symmetric pairwise distance matrix of shape (21, 21).
        """

        # Number of nodes
        num_points = points.shape[0]
        
        # Initialize the distance matrix
        dist_matrix = np.zeros((num_points, num_points))
        
        # Calculate pairwise Euclidean distances manually
        for i in range(num_points):
            for j in range(i, num_points):  # Calculate only upper triangle
                dist = np.linalg.norm(points[i] - points[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist  # Ensure symmetry
        
        # Normalize by the maximum distance
        max_distance = np.max(dist_matrix)
        if max_distance > 0:
            normalized_dist_matrix = dist_matrix / max_distance
        else:
            normalized_dist_matrix = dist_matrix  # If max distance is 0, avoid division

        return normalized_dist_matrix


class RotationFeatures(BaseFeatures):
    def __init__(self, points):
        super().__init__()
    def calculate_quaternions(keypoints):
        """
        Given a (21, 3) matrix of keypoints in 3D, calculate the quaternion representing
        the rotation for each segment between consecutive keypoints.
        
        Parameters:
        - keypoints: np.ndarray of shape (21, 3), representing 3D keypoints.
        
        Returns:
        - quaternions: List of quaternions (as [w, x, y, z] arrays) for each segment.
        """
        # List to store quaternions for each segment
        quaternions = []
        
        # Define a reference vector (e.g., pointing along x-axis)
        reference_vector = np.array([1, 0, 0])

        # Loop through each consecutive pair of keypoints
        for i in range(len(keypoints) - 1):
            # Calculate the direction vector for the current segment
            segment_vector = keypoints[i + 1] - keypoints[i]
            segment_vector /= np.linalg.norm(segment_vector)  # Normalize it

            # Calculate rotation quaternion from reference vector to segment vector

            rotation = R.from_rotvec(np.cross(reference_vector, segment_vector))
            quaternion = rotation.as_quat()  # Format: [x, y, z, w]
            
            # Reorder to [w, x, y, z] and append to list
            quaternions.append([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])

        return np.array(quaternions)
    


class HandFeatures(BaseFeatures):
    """Handles calculation of hand-related features."""
    WRIST_BASE = 0
    THUMB = [1, 2, 3, 4]
    INDEX = [5, 6, 7, 8]
    MIDDLE = [9, 10, 11, 12]
    RING = [13, 14, 15, 16]
    PINKY = [17, 18, 19, 20]
    INDEX_BASE, INDEX_TIP = INDEX[0], INDEX[-1]
    PINKY_BASE, PINKY_TIP = PINKY[0], PINKY[-1]
    THUMB_BASE, THUMB_TIP = THUMB[0], THUMB[-1]

    
    
    def __init__(self, pose_data, subsample_index = None, features = None, subsample_finger = None):
        super().__init__(pose_data)
        self.hamer_left = self.keypoints_hand_left
        self.normalized_hamer_left = self.normalized_keypoints_hand_left
        self.hamer_right = self.keypoints_hand_right
        self.normalized_hamer_right = self.normalized_keypoints_hand_right
        self.subsample_index = subsample_index
        self.subsample_finger = subsample_finger
        self.bones = [
            [0, 1], [1, 2], [2, 3], [3, 4],      # Thumb
            [0, 5], [5, 6], [6, 7], [7, 8],      # Index Finger
            [0, 9], [9, 10], [10, 11], [11, 12], # Middle Finger
            [0, 13], [13, 14], [14, 15], [15, 16], # Ring Finger
            [0, 17], [17, 18], [18, 19], [19, 20]  # Pinky Finger
        ]
        if subsample_index is not None:
            self.select_keypoints()

    def select_keypoints(self):
        indexes = self.subsample_index
        self.hamer_left = self.hamer_left[:, indexes, :]
        self.normalized_hamer_left = self.normalized_hamer_left[:, indexes, :]
        self.hamer_right = self.hamer_right[:, indexes, :]
        self.normalized_hamer_right = self.normalized_hamer_right[:, indexes, :]

        self.bones = [bone for bone in self.bones if bone[0] in indexes and bone[1] in indexes]
        
    
    def get_normals(self,kp1, kp2, kp3, data):
        return  [
            self.calculate_palm_normal(frame[self.WRIST_BASE], frame[self.INDEX_BASE], frame[self.PINKY_BASE])
            for frame in self.hamer_left
            ]
        
    
    def calculate_palm_normal(self, wrist, index_base, pinky_base):
        """Calculates the normal vector of the palm."""
        v1 = index_base - wrist
        v2 = pinky_base - wrist
        
        normal = np.cross(v1, v2)
        return normal / np.linalg.norm(normal)

    def calculate_finger_normal(self,finger_base, finger_tip):
        """Calculates the normal vector of a finger."""
        direction = finger_tip - finger_base
        return direction / np.linalg.norm(direction)

    def calculate_angular_difference(self,v1, v2):
        """Calculates the angular difference (in degrees) between two vectors."""
        v1 = np.array(v1)
        v2 = np.array(v2)
        
        # Normalize vectors
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        # Calculate dot product and angular difference
        dot_product = np.dot(v1_norm, v2_norm)
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Clip to avoid numerical errors
        angular_difference = np.degrees(angle_rad)
        
        # Calculate azimuth and elevation for each vector
        def spherical_coordinates(vector):
            x, y, z = vector
            azimuth = np.degrees(np.arctan2(y, x))
            elevation = np.degrees(np.arcsin(z / np.linalg.norm(vector)))
            return azimuth, elevation
        
        azimuth1, elevation1 = spherical_coordinates(v1)
        azimuth2, elevation2 = spherical_coordinates(v2)
        azimuth_diff = azimuth2 - azimuth1
        elevation_diff = elevation2 - elevation1
        return azimuth_diff, elevation_diff

    def est_orientation_features(self,keypoints, palm_normals = None):
        """
        Processes a sequence of keypoints to calculate palm normals, 
        finger normals, and angular differences.
        """
        T = len(keypoints)
        features = []
        reference_joint_index = -1

        for t in range(T):
            frame = keypoints[t]
            finger_orientations = []
            if palm_normals is not None:
                palm_normal = palm_normals[t]
            else:
                wrist = frame[self.WRIST_BASE]
                index_base = frame[self.INDEX_BASE]
                pinky_base = frame[self.PINKY_BASE]
                palm_normal = self.calculate_palm_normal(wrist, index_base, pinky_base)

            # Calculate finger normals
            finger_normals = {}
            for name, finger in [('thumb', HandFeatures.THUMB), 
                                 ('index', HandFeatures.INDEX), 
                                 ('middle', HandFeatures.MIDDLE), 
                                 ('ring', HandFeatures.RING), 
                                 ('pinky', HandFeatures.PINKY)]:
                finger_normal = self.calculate_finger_normal(finger_base = frame[finger[0]], finger_tip = frame[finger[reference_joint_index]])
                finger_normals[name] = finger_normal

            # Calculate angular differences between palm normal and finger normals
            angular_differences = {}
            for name, finger_normal in finger_normals.items():
                angular_differences[name] = self.calculate_angular_difference(palm_normal, finger_normal)
            
            for finger, value in angular_differences.items():
                if finger in self.subsample_finger:
                    finger_orientations.append([value[0], value[1]])
            features.append(finger_orientations)
        return np.array(features)