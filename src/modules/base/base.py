import numpy as np
import os
import pickle
import cv2
import traceback
from pose_format import Pose
from PoseTools.src.utils.preprocessing import PoseSelect, PoseNormalize
from PoseTools.src.modules.features.feature_transformations import HandFeatures, MaskFeatures

class DataLoader:
    """Handles loading of video, pose, and HAMER data."""
    # Handshape indexes
    WRIST_BASE = 0
    INDEX_BASE, INDEX_TIP = 5, 0
    PINKY_BASE, PINKY_TIP = 17, 0  
    THUMB_BASE, THUMB_TIP = 0, 0

    # Pose indexes
    SHOULDER_LEFT = 8
    SHOULDER_RIGHT = 9
    HIP_LEFT = 0
    HIP_RIGHT = 0
    WRIST_LEFT = 12
    WRIST_RIGHT = 13

    def __init__(self, base_filename, base_dir, args):
        # Define paths
        self.args = args
        self.base_filename = base_filename
        self.BASE_DIR = base_dir
        self.feature_transformations =  self.args.feature_transformation
        self.video_path = os.path.join(base_dir, 'video_files', f"{base_filename}.mp4")
        self.pose_path = os.path.join(base_dir, 'smplx', f"{base_filename}_b.npy")
        self.hamer_left_path = os.path.join(base_dir, 'hamer', f"{base_filename}", 'hamer_pkl', f"{base_filename}-L.pkl")
        self.hamer_right_path = os.path.join(base_dir, 'hamer', f"{base_filename}", 'hamer_pkl', f"{base_filename}-R.pkl")
        self.normalized_hamer_left_path = os.path.join(base_dir, 'hamer', f"{base_filename}", 'hamer_pkl', f"normalized_{base_filename}-L.pkl")
        self.normalized_hamer_right_path = os.path.join(base_dir, 'hamer', f"{base_filename}", 'hamer_pkl', f"normalized_{base_filename}-R.pkl")
        
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
            print('Loading reference poses... \n')
            reference_pose_path = '/home/gomer/oline/PoseTools/src/modules/handshapes/utils/build_references/references/reference_poses.pkl'
        elif self.feature_transformations == 'pdm':
            print('Loading pdm reference poses... \n')
            reference_pose_path = '/home/gomer/oline/PoseTools/src/modules/handshapes/utils/build_references/references/reference_pdms.pkl'

        with open(reference_pose_path, 'rb') as file:
            self.reference_poses = pickle.load(file)
            if self.args.mask_type is not None:
                masker = MaskFeatures(self.args.mask_type)
                for key, value in self.reference_poses.items():
                    self.reference_poses[key] = [masker.mask_features(pdm) for pdm in value]
                        
        
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

    def load_pose_poseformat(self, n_dims = 3):
        with open(self.pose_path, "rb") as file:
            self.pose_data = Pose.read(file.read())
            self.pose = self.pose_data.body.data.data
    
    def load_pose(self, n_dims = 3):
        self.pose = np.load(self.pose_path).squeeze(1)
            
    def load_hamer(self, file_path):
        """Loads HAMER data for left or right hand."""
        with open(file_path, "rb") as file:
            hamer_data = pickle.load(file)
        return hamer_data.get('keypoints')
    
    def select_data(self, start, stop, skip=3, padding = 12, boolean_activity_arrays=None, sign_activity_arrays=None, pre_cropped = False):
        
        if start > padding:
                start -= padding
        if stop + padding < self.num_frames:
            stop += padding
        #if pre_cropped:
        #    start = 0
        #    stop = self.num_frames + 1
  
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
        self.normalized_keypoints_hamer_left = self.normalized_keypoints_hamer_left[start:stop:skip]
        self.normalized_keypoints_hamer_right = self.normalized_keypoints_hamer_right[start:stop:skip]

        self.frames = self.frames[start:stop:skip]
        self.num_frames = len(self.pose)
        print(f"Selected frames: {self.num_frames}")

        self.velocity_left = np.diff(self.wrist_left, prepend=self.wrist_left[0])
        self.velocity_right = np.diff(self.wrist_right, prepend=self.wrist_right[0])

        
        boolean_activity_arrays = boolean_activity_arrays[0][start:stop:skip],  boolean_activity_arrays[1][start:stop:skip] 

        sign_activity_arrays = sign_activity_arrays[0][start:stop:skip], sign_activity_arrays[1][start:stop:skip]

        #if pre_cropped:
        #    L = True if np.sum(boolean_activity_arrays[0]) != 0 else False
        #    R = True if np.sum(boolean_activity_arrays[1]) != 0 else False
        #    boolean_L = [1 for i in range(self.num_frames)] if L else [0 for i in range(self.num_frames)]
        #    boolean_R = [1 for i in range(self.num_frames)] if R else [0 for i in range(self.num_frames)]
        #    sign_L = ['Active' for i in range(self.num_frames)] if L else ['Inactive' for i in range(self.num_frames)]
        #    sign_R = ['Active' for i in range(self.num_frames)] if R else ['Inactive' for i in range(self.num_frames)]
        #    boolean_activity_arrays = boolean_L, boolean_R
        #    sign_activity_arrays = sign_L, sign_R

        self.get_data_dict()
        return boolean_activity_arrays, sign_activity_arrays

    def get_data_dict(self):
        self.data_dict = {
            'keypoints_body': self.pose,
            'keypoints_hand_left': self.hamer_left,
            'normalized_keypoints_hand_left': self.normalized_hamer_left,
            'keypoints_hand_right': self.hamer_right,
            'normalized_keypoints_hand_right': self.normalized_hamer_right,
        }

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
        #self.pose = self.selector.clean_keypoints(self.pose)
        #self.pose = self.selector.get_keypoints_pose(self.pose)
        
        self.wrist_left = self.selector.get_left_wrist(self.pose)[:, :self.n_dims]
        self.wrist_right = self.selector.get_right_wrist(self.pose)[:, :self.n_dims]

        # TODO: The normalization separately off the pose and wrist is a little wierd
        #self.pose = self.normalizer.fullpose_normalization(self.pose)
        self.shoulder_left = self.pose[:, self.SHOULDER_LEFT, :self.n_dims]
        self.shoulder_right = self.pose[:, self.SHOULDER_RIGHT, :self.n_dims]
        self.hip_left = self.pose[:, self.HIP_LEFT, :self.n_dims]
        self.hip_right = self.pose[:, self.HIP_RIGHT, :self.n_dims]

        return self.pose, self.wrist_left, self.wrist_right, self.shoulder_left, self.shoulder_right, self.hip_left, self.hip_right

    def normalize_wrist_data(self, wrist_left, wrist_right):
        wrist_left, wrist_right = self.normalizer.reference_normalization(wrist_left, wrist_right)
        return self.normalizer.minmax_normalize_together(wrist_left, wrist_right)

    
class DataModule(DataLoader):
    """Main module that orchestrates data loading, preprocessing, and feature computation."""

    def __init__(self, base_filename, base_dir,args):
        super().__init__(base_filename, base_dir, args)
        self.subsample_index = args.subsample_index 
        self.feature_transformation = args.feature_transformation 
        self.subsample_finger = args.subsample_finger
        self.load_data()

        self.preprocess_data()
        
        self.get_data_dict()
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
        hand_features = HandFeatures(self.data_dict, self.subsample_index, self.feature_transformation, self.subsample_finger)
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
