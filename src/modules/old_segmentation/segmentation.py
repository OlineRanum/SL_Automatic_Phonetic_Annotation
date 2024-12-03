# pose_video_analyzer.py

import os
import cv2
import pickle
import numpy as np
from pose_format import Pose
from PoseTools.src.utils.preprocessing import PoseSelect, PoseNormalize
from PoseTools.src.modules.old_segmentation.segmentation_plot import WristMovementAnimator
import ruptures as rpt
from hmmlearn import hmm
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt

def smooth_binary_array(arr, min_run=10):
    """
    Smooth a binary numpy array by replacing runs of consecutive 0s or 1s
    that are shorter than min_run with the surrounding value.

    Parameters:
    - arr (np.ndarray): 1D numpy array of 0s and 1s.
    - min_run (int): Minimum number of consecutive identical values to keep.

    Returns:
    - np.ndarray: Smoothed numpy array.
    """
    if arr.ndim != 1:
        raise ValueError("Input array must be one-dimensional.")
    
    n = len(arr)
    if n == 0:
        return arr.copy()
    
    # Find the indices where the value changes
    diff = np.diff(arr)
    change_indices = np.where(diff != 0)[0] + 1

    # Append start and end indices
    run_starts = np.concatenate(([0], change_indices))
    run_ends = np.concatenate((change_indices, [n]))
    run_lengths = run_ends - run_starts
    run_values = arr[run_starts]

    # Identify runs that need to be smoothed
    small_runs = run_lengths < min_run
    if not np.any(small_runs):
        return arr.copy()  # No smoothing needed

    # Determine the replacement values for small runs
    # Initialize replacement values with the same run values
    replacements = run_values.copy()

    # For each small run, determine the replacement value based on neighbors
    # Shift run_values to get previous and next run values
    prev_values = np.roll(run_values, 1)
    next_values = np.roll(run_values, -1)

    # Handle edge cases
    prev_values[0] = run_values[0]  # First run has no previous
    next_values[-1] = run_values[-1]  # Last run has no next

    # If both neighbors are the same, use that value
    same_neighbors = (prev_values == next_values)
    replacements[small_runs & same_neighbors] = prev_values[small_runs & same_neighbors]

    # If neighbors differ, prefer the previous value (can be adjusted as needed)
    replacements[small_runs & ~same_neighbors] = prev_values[small_runs & ~same_neighbors]

    # Now, map the replacements back to the original array
    # Only replace the runs that are small
    small_run_indices = np.where(small_runs)[0]
    for idx in small_run_indices:
        start, end = run_starts[idx], run_ends[idx]
        smoothed_value = replacements[idx]
        arr[start:end] = smoothed_value

    return arr

class PoseVideoAnalyzer:
    BASE_DIR = '/home/gomer/oline/PoseTools/src/modules/demo/demo_files/sentences'

    VIDEO_PATH = os.path.join(BASE_DIR, 'video_files')
    POSE_PATH = os.path.join(BASE_DIR, 'pose_files')
    HAMER_PATH = os.path.join(BASE_DIR, 'hamer_pkl')

    def __init__(self, base_filename, BASE_DIR=None):
        if BASE_DIR is not None:
            BASE_DIR = BASE_DIR

        self.video_path = os.path.join(self.VIDEO_PATH, f"{base_filename}.mp4")
        if 'normalized_' in self.video_path:
            self.video_path = self.video_path.replace('normalized_', '')

        self.pose_path = os.path.join(self.POSE_PATH, f"{base_filename}.pose")
        self.hamer_left_path = os.path.join(self.HAMER_PATH, f"normalized_{base_filename}-L.pkl")
        self.hamer_right_path = os.path.join(self.HAMER_PATH, f"normalized_{base_filename}-R.pkl")
        
        self.load_pose()

        self.hamer_right = self.load_hamer("right")
        self.hamer_left = self.load_hamer("left")


    def load_pose(self, n_dims = 2):
        with open(self.pose_path, "rb") as file:
            self.pose_data = Pose.read(file.read())
            
        selector = PoseSelect("mediapipe_holistic_minimal_27")
        self.pose = selector.clean_keypoints(self.pose_data.body.data.data)
        self.pose = selector.get_keypoints_pose(self.pose)
        self.left_wrist = selector.get_left_wrist(self.pose)[:, :n_dims]
        self.right_wrist = selector.get_right_wrist(self.pose)[:, :n_dims]
        
        return self.pose, self.pose_data.body.confidence
    
    def load_hamer(self, side):
        if side == "left":
            with open(self.hamer_left_path, "rb") as file:
                hamer_data = pickle.load(file)
            return hamer_data.get('keypoints')
        elif side == "right":
            with open(self.hamer_right_path, "rb") as file:
                hamer_data = pickle.load(file)
            return hamer_data.get('keypoints')
        else:
            print(f"Invalid side: {side}")
            return None

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
    
    def train_hmm_with_best_initialization(self, features, n_components=2, covariance_type="diag", n_iter=300, tol=1e-3, n_initializations=5):
        best_score = -np.inf
        best_model = None

        for i in range(n_initializations):
            try:
                model = hmm.GaussianHMM(n_components=n_components,
                                        covariance_type=covariance_type,
                                        n_iter=n_iter,
                                        tol=tol,
                                        random_state=i,
                                        verbose=False)
                model.fit(features)
                score = model.score(features)
                #print(f"Initialization {i}: Score = {score}")
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                print(f"Initialization {i} failed: {e}")

        if best_model:
            #print(f"Best model score: {best_score}")
            return best_model
        else:
            raise ValueError("All HMM initializations failed.")

    def interpret_states(self, model, hidden_states):
        state_means = model.means_.mean(axis=1)
        movement_state = np.argmax(state_means)
        movement_labels = np.where(hidden_states == movement_state, 1, 0)
        return movement_labels

    def find_movement_segments(self, movement_labels):
        movement_indices = np.where(movement_labels == 1)[0]
        if movement_indices.size == 0:
            start_frame =  len(movement_labels) - 2
            end_frame = len(movement_labels) - 1
        else:
            start_frame = movement_indices[0]
            end_frame = movement_indices[-1]
        if start_frame == 0 and end_frame == len(movement_labels) - 1:
            start_frame =  len(movement_labels) - 2
            end_frame = len(movement_labels) - 1
        return start_frame, end_frame

    def detect_movement_frames(self, plot = True):
        # Preprocessing steps
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        left_wrist = np.expand_dims(self.left_wrist, 1)
        right_wrist = np.expand_dims(self.right_wrist, 1)
        left_wrist_vel = np.diff(left_wrist, axis=0)  # Shape: [T-1, dims]
        left_wrist_vel = np.vstack((left_wrist_vel, np.zeros((1, left_wrist_vel.shape[1]))))  
        right_wrist_smoothed = np.diff(right_wrist, axis=0)  # Shape: [T-1, dims]
        right_wrist_smoothed = np.vstack((right_wrist_smoothed, np.zeros((1, right_wrist_smoothed.shape[1]))))


        # PCA on handshape features
        pca = PCA(n_components=1)
        handshape_left_flat = self.hamer_left.reshape(self.num_frames, -1)
        handshape_right_flat = self.hamer_right.reshape(self.num_frames, -1)

        handshape_features_left = np.abs(pca.fit_transform(handshape_left_flat))
        handshape_features_right = np.abs(pca.fit_transform(handshape_right_flat))

        # Scale PCA features
        handshape_left_scaled = scaler.fit_transform(handshape_features_left)
        handshape_right_scaled = scaler.fit_transform(handshape_features_right)

        # Combine features
        combined_features_left = np.hstack((handshape_left_scaled, left_wrist, left_wrist_vel))
        combined_features_right = np.hstack((handshape_right_scaled, right_wrist, right_wrist_smoothed))
        joint_features = np.vstack((combined_features_left, combined_features_right))   
        model = self.train_hmm_with_best_initialization(joint_features,
                                                n_components=2,
                                                covariance_type="diag",
                                                n_iter=300,
                                                tol=1e-1,
                                                n_initializations=5)
        
        hidden_states = model.predict(joint_features)
        
        movement_labels = self.interpret_states(model, hidden_states)
        self.movement_labels_left, self.movement_labels_right = smooth_binary_array(movement_labels[:self.num_frames]), smooth_binary_array(movement_labels[self.num_frames:])
        
        print(self.movement_labels_left)
        print(len(self.movement_labels_left))
        # Identify movement segments
        left_start_frame, left_end_frame = self.find_movement_segments(self.movement_labels_left)
        right_start_frame, right_end_frame = self.find_movement_segments(self.movement_labels_right)

        # Print results
        if left_start_frame is None:
            print("No movement detected in left hand")
        else:
            print(f"\nLeft hand movement starts at frame {left_start_frame//3}")
            print(f"Left hand movement ends at frame {left_end_frame//3}\n")

        if right_start_frame is None:
            print("No movement detected in right hand")
        else:
            print(f"Right hand movement starts at frame {right_start_frame//3}")
            print(f"Right hand movement ends at frame {right_end_frame//3}")

        # Plot hidden states
        if plot:
            self.plot_hidden_states(self.movement_labels_left, self.movement_labels_right)

        # Return frames
        start_frame = [left_start_frame, right_start_frame]
        end_frame = [left_end_frame, right_end_frame]

        return start_frame, end_frame

    def plot_hidden_states(self, movement_labels_left, movement_labels_right):
        import matplotlib.pyplot as plt

        frames = np.arange(len(movement_labels_left))

        plt.figure(figsize=(15, 6))

        plt.subplot(2, 1, 1)
        plt.plot(frames, movement_labels_left, label='Left Hand Movement', color='blue')
        plt.fill_between(frames, 0, movement_labels_left, where=movement_labels_left==1, color='blue', alpha=0.3)
        plt.xlabel('Frame')
        plt.ylabel('Movement')
        plt.title('Left Hand Movement Detection')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(frames, movement_labels_right, label='Right Hand Movement', color='red')
        plt.fill_between(frames, 0, movement_labels_right, where=movement_labels_right==1, color='red', alpha=0.3)
        plt.xlabel('Frame')
        plt.ylabel('Movement')
        plt.title('Right Hand Movement Detection')
        plt.legend()

        plt.tight_layout()
        plt.show()


    def _find_movement_index(self, wrist_diff, threshold):
        """Helper function to find the first index exceeding the threshold."""
        movement_detected = wrist_diff > threshold
        return np.argmax(movement_detected) + 1 if np.any(movement_detected) else -1

    def normalize_wrist_data(self):
        normalizer = PoseNormalize(self.left_wrist, self.right_wrist, self.pose)
        self.left_wrist, self.right_wrist = normalizer.reference_normalization()
        self.left_wrist, self.right_wrist = normalizer.minmax_normalize_together()

    def print_frame_counts(self):
        counts = self.get_frame_counts()
        for key, count in counts.items():
            print(f"Number of frames in the {key} file: {count}")


    def construct_boolean(self, pose, first, last, mode = 'full'):
        if mode == 'full':  
            return self.movement_labels_left, self.movement_labels_right
        elif mode == 'start_stop':
            bool_L = np.zeros(len(pose))
            bool_R = np.zeros(len(pose))
            bool_L[first[0]:last[0]] = 1
            bool_R[first[1]:last[1]] = 1
            return bool_L, bool_R


def main_activation(base_filename, create_anim=False, save_anim_path=None, handshapes=None, handedness=None, orientations = None, locations = None, fps = 5):
    analyzer = PoseVideoAnalyzer(base_filename)
    analyzer.print_frame_counts()
    analyzer.normalize_wrist_data()
    first, last = analyzer.detect_movement_frames()
    
    if create_anim:
        if save_anim_path:
            animator = WristMovementAnimator(analyzer.BASE_DIR, analyzer.left_wrist, analyzer.movement_labels_left, analyzer.right_wrist, analyzer.movement_labels_right, base_filename, handshapes = handshapes, handedness = handedness, orientations = orientations, locations = locations)
            animator.create_animation(save_path=save_anim_path, first_movement=first, last_movement=last, fps=fps)
    return analyzer.construct_boolean(analyzer.pose, first, last)


if __name__ == "__main__":
    base_filename = "M20241107_6254"
    gif_path = '/home/gomer/oline/PoseTools/src/modules/demo/graphics/gifs/wrist_movement_animation.gif'
    main_activation(base_filename, create_anim=True, save_anim_path=gif_path)
