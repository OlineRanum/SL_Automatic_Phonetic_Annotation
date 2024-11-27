import numpy as np
import os 
from pose_format import Pose
from PoseTools.src.utils.preprocessing import PoseSelect, PoseNormalize
import pickle
import cv2
import traceback

class DataModule:
    BASE_DIR = '/home/gomer/oline/PoseTools/src/modules/demo/demo_files/animations'

    # Datapaths
    VIDEO_PATH = os.path.join(BASE_DIR, 'video_files')
    POSE_PATH = os.path.join(BASE_DIR, 'pose_files')
    HAMER_PATH = os.path.join(BASE_DIR, 'hamer_pkl')

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
    

    def __init__(self, base_filename, BASE_DIR=None):
        if BASE_DIR is not None:
            BASE_DIR = BASE_DIR
        
        self.base_filename = base_filename

        self.video_path = os.path.join(self.VIDEO_PATH, f"{base_filename}.mp4")
        if 'normalized_' in self.video_path:
            self.video_path = self.video_path.replace('normalized_', '')

        self.pose_path = os.path.join(self.POSE_PATH, f"{base_filename}.pose")
        self.hamer_left_path = os.path.join(self.HAMER_PATH, f"{base_filename}-L.pkl")
        self.hamer_right_path = os.path.join(self.HAMER_PATH, f"{base_filename}-R.pkl")
        
        self.normalized_hamer_left_path = os.path.join(self.HAMER_PATH, f"normalized_{base_filename}-L.pkl")
        self.normalized_hamer_right_path = os.path.join(self.HAMER_PATH, f"normalized_{base_filename}-R.pkl")
        
        self.load_pose()
        self.hamer_right, self.normalized_hamer_right = self.load_hamer("right")
        self.hamer_left, self.normalized_hamer_left = self.load_hamer("left")

        self.normalize_wrist_data()
        
        self.print_frame_counts()

        self.frames = []
        self.load_frames()
        self.num_frames = len(self.frames)

        self.normal_vectors_right = np.array([
                    self.calculate_palm_normal(frame[self.WRIST_BASE], frame[self.INDEX_BASE], frame[self.PINKY_BASE])
                    for frame in self.hamer_right
                ])
        
        self.normal_vectors_left = np.array([
                    self.calculate_palm_normal(frame[self.WRIST_BASE], frame[self.INDEX_BASE], frame[self.PINKY_BASE])
                    for frame in self.hamer_left
                ])
    
    def load_frames(self):
        if not self.video_path or not os.path.exists(self.video_path):
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
                if frame is None:
                    continue
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
            
        selector = PoseSelect("mediapipe_holistic_minimal_27")
        self.pose = selector.clean_keypoints(self.pose_data.body.data.data)
        self.pose = selector.get_keypoints_pose(self.pose)
        
        normalizer = PoseNormalize(None, None, self.pose)
        
        self.wrist_left = selector.get_left_wrist(self.pose)[:, :n_dims]
        self.wrist_right = selector.get_right_wrist(self.pose)[:, :n_dims]

        # TODO: The normalization separately off the pose and wrist is a little wierd
        self.pose = normalizer.fullpose_normalization()
        self.shoulder_left = self.pose[:, self.SHOULDER_LEFT, :n_dims]
        self.shoulder_right = self.pose[:, self.SHOULDER_RIGHT, :n_dims]
        self.hip_left = self.pose[:, self.HIP_LEFT, :n_dims]
        self.hip_right = self.pose[:, self.HIP_RIGHT, :n_dims]



        return self.pose, self.pose_data.body.confidence
        
    def load_hamer(self, side):
        if side == "left":
            with open(self.hamer_left_path, "rb") as file:
                hamer_data = pickle.load(file)
            with open(self.normalized_hamer_left_path, "rb") as file:
                normalized_hamer_data = pickle.load(file)
        elif side == "right":
            with open(self.hamer_right_path, "rb") as file:
                hamer_data = pickle.load(file)
            with open(self.normalized_hamer_right_path, "rb") as file:
                normalized_hamer_data = pickle.load(file)
        else:
            print(f"Invalid side: {side}")
            return None
        return hamer_data.get('keypoints'), normalized_hamer_data.get('keypoints')

        
    def load_video_frame_count(self):
        """Load video and return total frame count."""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"Error opening video file {self.video_path}")
            return 0

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in video: {frame_count}")
        cap.release()
        return frame_count

    def get_hamer_frame_count(self, side):
        if side == "left":
            return len(self.hamer_left)
        elif side == "right":   
            return len(self.hamer_right)
        
    def get_pose_frame_count(self):
        return len(self.pose)

    def get_frame_counts(self):
        self.num_frames = self.load_video_frame_count()
        return {
            "pose": self.get_pose_frame_count() if self.pose is not None else 0,
            "video": self.num_frames,
            "hamer_left": self.get_hamer_frame_count('left'),
            "hamer_right": self.get_hamer_frame_count('right'),
            }

    def normalize_wrist_data(self):
        normalizer = PoseNormalize(self.wrist_left, self.wrist_right, self.pose)
        self.wrist_left, self.wrist_right = normalizer.reference_normalization()
        self.wrist_left, self.wrist_right = normalizer.minmax_normalize_together()

    def print_frame_counts(self):
        counts = self.get_frame_counts()
        for key, count in counts.items():
            print(f"Number of frames in the {key} file: {count}")


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

        boolean_activity_arrays = boolean_activity_arrays[0][start:stop:3],  boolean_activity_arrays[1][start:stop:3] 
        sign_activity_arrays = sign_activity_arrays[0][start:stop:3], sign_activity_arrays[1][start:stop:3]
        
        return boolean_activity_arrays, sign_activity_arrays
    

    def calculate_palm_normal(self, wrist, index_base, pinky_base):
        # Calculate vectors
        v1 = index_base - wrist
        v2 = pinky_base - wrist
        
        # Cross product to find the normal vector
        normal = np.cross(v1, v2)
        
        # Normalize the normal vector
        normal_unit = normal / np.linalg.norm(normal)
        
        return normal_unit