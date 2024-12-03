import numpy as np
import os
import pickle
import cv2
import traceback
from pose_format import Pose
from PoseTools.src.utils.preprocessing import PoseSelect, PoseNormalize


class DataLoader:
    """Handles loading of video, pose, and HAMER data."""
    # Handshape indexes
    WRIST_BASE = 0
    INDEX_BASE, INDEX_TIP = 5, 0
    PINKY_BASE, PINKY_TIP = 17, 0  
    THUMB_BASE, THUMB_TIP = 0, 0

    # Pose indexes
    SHOULDER_LEFT = 11
    SHOULDER_RIGHT = 12
    HIP_LEFT = 23
    HIP_RIGHT = 24
    WRIST_LEFT = 15
    WRIST_RIGHT = 16

    def __init__(self, base_filename, base_dir, feature_transformations = None):
        # Define paths
        self.base_filename = base_filename
        self.BASE_DIR = base_dir
        self.feature_transformations = feature_transformations
        self.video_path = os.path.join(base_dir, 'video_files', f"{base_filename}.mp4")
        self.pose_path = os.path.join(base_dir, 'pose_files', f"{base_filename}.pose")
        self.hamer_left_path = os.path.join(base_dir, 'hamer_pkl', f"{base_filename}-L.pkl")
        self.hamer_right_path = os.path.join(base_dir, 'hamer_pkl', f"{base_filename}-R.pkl")
        self.normalized_hamer_left_path = os.path.join(base_dir, 'hamer_pkl', f"normalized_{base_filename}-L.pkl")
        self.normalized_hamer_right_path = os.path.join(base_dir, 'hamer_pkl', f"normalized_{base_filename}-R.pkl")
        
        # Initiate data components 
        self.frames = []
        self.pose_data = None
        self.hamer_left = None
        self.normalized_hamer_left = None
        self.hamer_right = None
        self.normalized_hamer_right = None
        self.num_frames = 0

    
    def load_reference_poses(self):
        # Load reference poses
        if self.feature_transformations == 'orientation':
            reference_pose_path = '/home/gomer/oline/PoseTools/src/modules/handshapes/utils/build_references/references/reference_poses.pkl'
        else:
            reference_pose_path = '/home/gomer/oline/PoseTools/src/modules/handshapes/utils/build_references/references/reference_pdms.pkl'
        with open(reference_pose_path, 'rb') as file:
            self.reference_poses = pickle.load(file)
        
        
        print(f"Reference poses loaded: {list(self.reference_poses.keys())}")
        print(f"Number of reference poses: {len(list(self.reference_poses.keys()))}")

        

    def load_data(self):
        """Loads all data components."""
        self.load_frames()
        self.frames = self.frames
        self.load_pose()
        self.hamer_left = self.load_hamer(self.hamer_left_path)
        self.normalized_hamer_left = self.load_hamer(self.normalized_hamer_left_path)
        self.normalized_keypoints_hamer_left = self.normalized_hamer_left.copy()
        self.hamer_right = self.load_hamer(self.hamer_right_path)
        self.normalized_hamer_right = self.load_hamer(self.normalized_hamer_right_path)
        self.normalized_keypoints_hamer_right = self.normalized_hamer_right.copy()        
        self.num_frames = len(self.frames)
        self.load_reference_poses()

    def load_frames(self):
        """Loads frames from video."""
        if not os.path.exists(self.video_path):
            print(f"Video file does not exist at: {self.video_path}")
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {self.video_path}")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.frames.append(frame_rgb)
            cap.release()
            print(f"Total frames captured: {len(self.frames)}")
        except Exception as e:
            print(f"Error during frame capture: {e}")
            traceback.print_exc()
            cap.release()

    def load_pose(self, n_dims = 3):
        with open(self.pose_path, "rb") as file:
            self.pose_data = Pose.read(file.read())
            self.pose = self.pose_data.body.data.data
        
            
        
    def load_hamer(self, file_path):
        """Loads HAMER data for left or right hand."""
        with open(file_path, "rb") as file:
            hamer_data = pickle.load(file)
        return hamer_data.get('keypoints')
    
    def select_data(self, start, stop, skip=3, padding = 12, boolean_activity_arrays=None, sign_activity_arrays=None):
        
        if start > padding:
            start -= padding


        if stop + padding < self.num_frames:
            stop += padding
  
        self.pose = self.pose[start:stop:skip]

        self.wrist_left = self.wrist_left[start:stop:skip]
        self.wrist_right = self.wrist_right[start:stop:skip]
        self.shoulder_left = self.shoulder_left[start:stop:skip]
        self.shoulder_right = self.shoulder_right[start:stop:skip]
        self.hip_left = self.hip_left[start:stop:skip]
        self.hip_right = self.hip_right[start:stop:skip]
        self.hamer_left = self.hamer_left[start:stop:skip]
        self.hamer_right = self.hamer_right[start:stop:skip]
        self.normalized_hamer_left = self.normalized_hamer_left[start:stop:skip]
        self.normalized_hamer_right = self.normalized_hamer_right[start:stop:skip]
        self.normal_vectors_left = self.normal_vectors_left[start:stop:skip]
        self.normal_vectors_right = self.normal_vectors_right[start:stop:skip]

        self.frames = self.frames[start:stop:skip]
        self.num_frames = len(self.pose)
        print(f"Selected frames: {self.num_frames}")

        self.velocity_left = np.diff(self.wrist_left, prepend=self.wrist_left[0])
        self.velocity_right = np.diff(self.wrist_right, prepend=self.wrist_right[0])

        boolean_activity_arrays = boolean_activity_arrays[0][start:stop:skip],  boolean_activity_arrays[1][start:stop:skip] 
        sign_activity_arrays = sign_activity_arrays[0][start:stop:skip], sign_activity_arrays[1][start:stop:skip]
        
        return boolean_activity_arrays, sign_activity_arrays


class Preprocessor:
    """Handles normalization and preprocessing of data."""
        # Pose indexes
    SHOULDER_LEFT = 11
    SHOULDER_RIGHT = 12
    HIP_LEFT = 23
    HIP_RIGHT = 24
    WRIST_LEFT = 15
    WRIST_RIGHT = 16

    def __init__(self, pose, n_dims=3, num_frames=None):
        self.num_frames = num_frames
        print(f"Total frames: {self.num_frames}")
        self.pose = pose
        self.n_dims = n_dims
        self.selector = PoseSelect("mediapipe_holistic_minimal_27")
        self.normalizer = PoseNormalize(None, None, self.pose)
        
        

    def normalize_pose(self):
        self.pose = self.selector.clean_keypoints(self.pose)
        self.pose = self.selector.get_keypoints_pose(self.pose)
        
        self.wrist_left = self.selector.get_left_wrist(self.pose)[:, :self.n_dims]
        self.wrist_right = self.selector.get_right_wrist(self.pose)[:, :self.n_dims]

        # TODO: The normalization separately off the pose and wrist is a little wierd
        self.pose = self.normalizer.fullpose_normalization(self.pose)
        self.shoulder_left = self.pose[:, self.SHOULDER_LEFT, :self.n_dims]
        self.shoulder_right = self.pose[:, self.SHOULDER_RIGHT, :self.n_dims]
        self.hip_left = self.pose[:, self.HIP_LEFT, :self.n_dims]
        self.hip_right = self.pose[:, self.HIP_RIGHT, :self.n_dims]

        return self.pose, self.wrist_left, self.wrist_right, self.shoulder_left, self.shoulder_right, self.hip_left, self.hip_right

    def normalize_wrist_data(self, wrist_left, wrist_right):
        wrist_left, wrist_right = self.normalizer.reference_normalization(wrist_left, wrist_right)
        return self.normalizer.minmax_normalize_together(wrist_left, wrist_right)
    
    


import numpy as np

class HandFeatures:
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

    
    
    def __init__(self, hamer_left, normalized_hamer_left, hamer_right, normalized_hamer_right, subsample_index = None, features = None, subsample_finger = None):
        self.hamer_left = hamer_left
        self.normalized_hamer_left = normalized_hamer_left
        self.hamer_right = hamer_right
        self.normalized_hamer_right = normalized_hamer_right
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
    


class DataModule(DataLoader):
    """Main module that orchestrates data loading, preprocessing, and feature computation."""

    def __init__(self, base_filename, base_dir,args):
        super().__init__(base_filename, base_dir, args.feature_transformation)
        self.subsample_index = args.subsample_index 
        self.feature_transformation = args.feature_transformation 
        self.subsample_finger = args.subsample_finger
        self.load_data()

        self.preprocess_data()
        
        self.compute_features()
        self.print_summary()


    def preprocess_data(self):
        """Preprocesses the pose and wrist data."""
        self.preprocessor = Preprocessor(self.pose, num_frames=self.num_frames)
        pose_data = self.preprocessor.normalize_pose()
        self.pose, self.wrist_left, self.wrist_right, self.shoulder_left, self.shoulder_right, self.hip_left, self.hip_right = pose_data
        self.wrist_left, self.wrist_right = self.preprocessor.normalize_wrist_data(self.wrist_left, self.wrist_right)

    def compute_features(self):
        """Computes features like palm normals."""
        hand_features = HandFeatures(self.hamer_left, self.normalized_hamer_left, self.hamer_right, self.normalized_hamer_right, self.subsample_index, self.feature_transformation, self.subsample_finger)
        self.normal_vectors_left = hand_features.get_normals(self.WRIST_BASE, self.INDEX_BASE, self.PINKY_BASE, self.hamer_left)
        self.normal_vectors_right = hand_features.get_normals(self.WRIST_BASE, self.INDEX_BASE, self.PINKY_BASE, self.hamer_left)

        if self.feature_transformation == 'orientation':
            
            self.hamer_left = hand_features.est_orientation_features(self.hamer_left)
            self.hamer_right = hand_features.est_orientation_features(self.hamer_right)
            self.normalized_hamer_left = hand_features.est_orientation_features(self.normalized_hamer_left)
            self.normalized_hamer_right = hand_features.est_orientation_features(self.normalized_hamer_right)
            new_reference_poses = {}
            
            
            for poses, val in self.reference_poses.items():
                values = []
                for pose in val:
                    
                    values.append(hand_features.est_orientation_features(pose[np.newaxis, :]))
                new_reference_poses[poses] = values
            self.reference_poses = new_reference_poses        
        #hand_features.process_sequence(self.hamer_left, self.normal_vectors_left)



    def print_summary(self):
        """Prints summary of loaded data."""
        print(f"Total frames: {len(self.frames)}")
        print(f"Pose data frames: {len(self.pose)}")
        print(f"Left HAMER frames: {len(self.hamer_left)}")
        print(f"Right HAMER frames: {len(self.hamer_right)}")




# Example usage
if __name__ == "__main__":
    base_dir = '/home/gomer/oline/PoseTools/data/demo_files/sentences'
    base_filename = 'example_file'
    data_module = DataModule(base_filename, base_dir)
    data_module.load_data()
    data_module.preprocess_data()
    data_module.compute_features()
    data_module.print_summary()
