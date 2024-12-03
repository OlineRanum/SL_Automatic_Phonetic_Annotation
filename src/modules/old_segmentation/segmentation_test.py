# pose_video_analyzer.py

import os
import cv2
import pickle
import numpy as np
from pose_format import Pose
from PoseTools.src.utils.preprocessing import PoseSelect, PoseNormalize
from PoseTools.src.modules.segmentation.segmentation_plot import WristMovementAnimator
import ruptures as rpt

class PoseVideoAnalyzer:
    # BASE_DIR = '/home/gomer/fonoProc/sentences'



    def __init__(self, base_filename, BASE_DIR=None):
        
        #determine if there is filepath in base_filename, then extract it
        if '/' in base_filename:
            base_filename = base_filename.split('/')[-1]
            
        #then we get base_dir from base_filename
        if BASE_DIR is None:
            BASE_DIR = os.path.dirname(base_filename)
            
        VIDEO_PATH = os.path.join(BASE_DIR, 'video_files')
        POSE_PATH = os.path.join(BASE_DIR, 'pose_files')
        HAMER_PATH = os.path.join(BASE_DIR, 'hamer_pkl')

        self.video_path = os.path.join(VIDEO_PATH, f"{base_filename}.mp4")
        if 'normalized_' in self.video_path:
            self.video_path = self.video_path.replace('normalized_', '')

        self.pose_path = os.path.join(POSE_PATH, f"{base_filename}.pose")
        self.hamer_left_path = os.path.join(HAMER_PATH, f"normalized_{base_filename}-L.pkl")
        self.hamer_right_path = os.path.join(HAMER_PATH, f"normalized_{base_filename}-R.pkl")
        #remove .mp4 from hamer path
        self.hamer_left_path = self.hamer_left_path.replace('.mp4', '')
        self.hamer_right_path = self.hamer_right_path.replace('.mp4', '')
        print(self.hamer_left_path)
        self.pose = None
        self.hamer_right = self.load_hamer("right")
        self.hamer_left = self.load_hamer("left")
        self.left_wrist = None
        self.right_wrist = None
        

    def load_pose(self):
        with open(self.pose_path, "rb") as file:
            self.pose_data = Pose.read(file.read())
            
        selector = PoseSelect("mediapipe_holistic_minimal_27")
        self.pose = selector.clean_keypoints(self.pose_data.body.data.data)
        self.pose = selector.get_keypoints_pose(self.pose)
        self.left_wrist = selector.get_left_wrist(self.pose)[:, :2]
        self.right_wrist = selector.get_right_wrist(self.pose)[:, :2]
        
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

    def load_hamer_frame_count(self, path):
        try:
            with open(path, "rb") as file:
                hamer_data = pickle.load(file)
            print()
            keypoints = hamer_data.get('keypoints')
            if keypoints is not None:
                return keypoints.shape[0]
            print(f"'keypoints' not found in {path}")
        except Exception as e:
            print(f"Error loading {path}: {e}")
        return 0

    def get_frame_counts(self):
        pose, _ = self.load_pose()
        return {
            "pose": pose.shape[0] if pose is not None else 0,
            "video": self.load_video_frame_count(),
            "hamer_left": self.load_hamer_frame_count(self.hamer_left_path),
            "hamer_right": self.load_hamer_frame_count(self.hamer_right_path),
            }
    

    def detect_movement_frames(self):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import MinMaxScaler
        import numpy as np
        from hmmlearn import hmm

        def butter_lowpass_filter(data, cutoff, fs, order=4):
            from scipy.signal import butter, filtfilt
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            y = filtfilt(b, a, data, axis=0)
            return y

        def fit_hmm(features, n_components=2):
            model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000, tol=0.5,verbose=True, random_state=42)
            model.fit(features)
            hidden_states = model.predict(features)
            return model, hidden_states

        def interpret_states(model, hidden_states):
            state_means = model.means_.mean(axis=1)
            movement_state = np.argmax(state_means)
            no_movement_state = np.argmin(state_means)
            movement_labels = np.where(hidden_states == movement_state, 1, 0)
            return movement_labels

        def find_movement_segments(movement_labels):
            movement_indices = np.where(movement_labels == 1)[0]
            if movement_indices.size == 0:
                return None, None
            start_frame = movement_indices[0]
            end_frame = movement_indices[-1]
            return start_frame, end_frame

        # Parameters
        fs = 60  # Frame rate (adjust as necessary)
        cutoff = 3  # Cutoff frequency based on noise characteristics
        window_size = 10  # Not used in HMM, but kept for consistency
        n_components = 2  # Number of hidden states for HMM

        # Ensure Wrist Data is 2D
        if self.left_wrist.ndim == 1:
            self.left_wrist = self.left_wrist.reshape(-1, 1)
        if self.right_wrist.ndim == 1:
            self.right_wrist = self.right_wrist.reshape(-1, 1)

        # Filter the wrist positions (assuming already min-max normalized)
        left_wrist_filtered = self.left_wrist
        right_wrist_filtered = self.right_wrist

        num_frames = len(left_wrist_filtered)

        # Apply PCA to handshape data separately for left and right
        pca_left = PCA(n_components=1)  # Adjust as needed
        handshape_left_flat = self.hamer_left.reshape(num_frames, -1)
        handshape_features_left = pca_left.fit_transform(handshape_left_flat)
        handshape_features_left = np.abs(handshape_features_left)

        pca_right = PCA(n_components=1)  # Adjust as needed
        handshape_right_flat = self.hamer_right.reshape(num_frames, -1)
        handshape_features_right = pca_right.fit_transform(handshape_right_flat)
        handshape_features_right = np.abs(handshape_features_right)

        # **Scale Handshape Features to [0, 1]**
        scaler_handshape_left = MinMaxScaler(feature_range=(0, 1))
        handshape_features_left = scaler_handshape_left.fit_transform(handshape_features_left)

        scaler_handshape_right = MinMaxScaler(feature_range=(0, 1))
        handshape_features_right = scaler_handshape_right.fit_transform(handshape_features_right)

        # **Combine with Wrist Positions**
        combined_features_left = np.concatenate((handshape_features_left, left_wrist_filtered), axis=1)
        combined_features_right = np.concatenate((handshape_features_right, right_wrist_filtered), axis=1)

        # Fit HMM for left hand
        model_left, hidden_states_left = fit_hmm(combined_features_left, n_components=n_components)
        movement_labels_left = interpret_states(model_left, hidden_states_left)
        left_start_frame, left_end_frame = find_movement_segments(movement_labels_left)
        if left_start_frame is None:
            print("No movement detected in left hand")
        else:
            print(f"Left hand movement starts at frame {left_start_frame //3}")
            print(f"Left hand movement ends at frame {left_end_frame//3}")

        # Fit HMM for right hand
        model_right, hidden_states_right = fit_hmm(combined_features_right, n_components=n_components)
        movement_labels_right = interpret_states(model_right, hidden_states_right)
        right_start_frame, right_end_frame = find_movement_segments(movement_labels_right)
        if right_start_frame is None:
            print("No movement detected in right hand")
        else:
            print(f"Right hand movement starts at frame {right_start_frame//3}")
            print(f"Right hand movement ends at frame {right_end_frame//3}")

        # Return frames
        start_frame = [left_start_frame, right_start_frame]
        end_frame = [left_end_frame, right_end_frame]
        return start_frame, end_frame

    '''
    def detect_movement_frames(self):
        
        def cumulative_movement(data, window_size):
            # Ensure data is a NumPy array
            data = np.asarray(data)
            
            # Compute cumulative sum per feature
            cumsum = np.cumsum(data, axis=0)
            
            # Pad the cumulative sum with zeros at the beginning
            padding = np.zeros((1, data.shape[1]))
            cumsum = np.vstack((padding, cumsum))
            
            # Compute the cumulative movement over the window size
            cumulative = cumsum[window_size:] - cumsum[:-window_size]
            
            # Normalize by window size to get average cumulative movement
            return cumulative / window_size
        
        num_frames = len(self.left_wrist)

        # Apply Pca
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)  # Retain top 20 principal components
        combined_data = np.concatenate(
            (self.hamer_left.reshape(num_frames, -1), self.hamer_right.reshape(num_frames, -1)),
            axis=0
        )
        print('combined_data', combined_data.shape) 
        # Apply PCA to the combined data
        pca = PCA(n_components=1)  # Adjust the number of components as needed
        combined_features = np.abs(pca.fit_transform(combined_data))
        print('combined_features', combined_features.shape)
        # Split the combined features back into left and right components
        # Determine the number of features for each hand (assuming they are equal)
        
        print(num_frames)
        handshape_features_left = combined_features[:num_frames].reshape(num_frames, -1)
        handshape_features_right = combined_features[num_frames:].reshape(num_frames, -1)
        print('handshape_features', handshape_features_left.shape)
        print('handshape_features', handshape_features_right.shape)
        # Combined features for left hand
        combined_features_left = np.concatenate((handshape_features_left, np.expand_dims(self.left_wrist, axis=1)), axis=1)
        combined_features_right = np.concatenate((handshape_features_right, np.expand_dims(self.right_wrist, axis=1)), axis=1)

        
        # Compute differences between consecutive frames
        diff_combined_left = np.diff(combined_features_left, axis=0)
        diff_combined_right = np.diff(combined_features_right, axis=0)

        print(diff_combined_left)

        # Compute cumulative movement per feature
        window_size = 10  # Adjust as needed

        left_movement_cumulative = cumulative_movement(diff_combined_left, window_size)
        right_movement_cumulative = cumulative_movement(diff_combined_right, window_size)

        # Compute movement magnitudes per timestep by taking the norm across features
        left_wrist_cumulative = np.linalg.norm(left_movement_cumulative, axis=1)
        right_wrist_cumulative = np.linalg.norm(right_movement_cumulative, axis=1)
        # Combine both arrays to compute global min and max
        combined = np.concatenate([left_wrist_cumulative, right_wrist_cumulative])

        # Compute global min and max
        global_min = np.min(combined)
        global_max = np.max(combined)

        # Normalize each component using the global min and max
        left_wrist_cumulative = (left_wrist_cumulative - global_min) / (global_max - global_min)
        right_wrist_cumulative = (right_wrist_cumulative - global_min) / (global_max - global_min)


        # Apply change point detection to left hand cumulative movement
        model = "l2"
        algo_left = rpt.Pelt(model=model).fit(left_wrist_cumulative)
        k_left = 0.03  # Adjust as needed
        penalty_left = k_left * np.std(left_wrist_cumulative) * np.log(len(left_wrist_cumulative))
        change_points_left = algo_left.predict(pen=penalty_left)

        # Apply change point detection to right hand cumulative movement
        algo_right = rpt.Pelt(model=model).fit(right_wrist_cumulative)
        k_right = 0.03  # Adjust as needed
        penalty_right = k_right * np.std(right_wrist_cumulative) * np.log(len(right_wrist_cumulative))
        change_points_right = algo_right.predict(pen=penalty_right)

        # Adjust indices due to windowing
        offset = window_size // 2

        # Interpret the change points for left hand
        print(np.array(change_points_left)//3)
        print(np.array(change_points_right)//3)
        print('features', handshape_features_left)
        import matplotlib.pyplot as plt
        plt.figure()
        #plt.plot(combined_features_left[:,0][::3], 'r')
        #plt.plot(combined_features_left[:,1][::3], 'b')
        plt.plot(left_wrist_cumulative[::3], 'k')
        print(self.hamer_left[0])

        #plt.plot(combined_features_right[:,0][::3], 'r') # Handshape
        #plt.plot(combined_features_right[:,1][::3], 'b') # Movement
        plt.plot(right_wrist_cumulative[::3], 'r')
        plt.title('right')
        plt.show()  
        if len(change_points_left) >= 2:
            left_start_frame = change_points_left[0] + offset
            left_end_frame = change_points_left[-2] + offset
        else:
            left_start_frame = None
            left_end_frame = None

        if left_start_frame is None:
            print("No movement detected in left hand")
        else:
            print(f"Left hand movement starts at frame {left_start_frame //3}")
            print(f"Left hand movement ends at frame {left_end_frame //3}")

        # Interpret the change points for right hand
        if len(change_points_right) >= 2:
            right_start_frame = change_points_right[0] + offset
            right_end_frame = change_points_right[-2] + offset
        else:
            right_start_frame = None
            right_end_frame = None

        if right_start_frame is None:
            print("No movement detected in right hand")
        else:
            print(f"Right hand movement starts at frame {right_start_frame//3}")
            print(f"Right hand movement ends at frame {right_end_frame//3}")

        start_frame = [left_start_frame, right_start_frame]
        end_frame = [left_end_frame, right_end_frame]
        return start_frame, end_frame
    
    def detect_movement_frame(self, threshold, sustained=False, reverse=False):
        # Determine the wrist data, reversed if specified
        left_wrist = self.left_wrist[::-1] if reverse else self.left_wrist
        right_wrist = self.right_wrist[::-1] if reverse else self.right_wrist

        # Calculate frame-to-frame absolute differences for left and right wrists
        left_wrist_diff = np.abs(np.diff(left_wrist))
        right_wrist_diff = np.abs(np.diff(right_wrist))

        # Find the first frame exceeding the threshold
        movement_left_idx = self._find_movement_index(left_wrist_diff, threshold)
        movement_right_idx = self._find_movement_index(right_wrist_diff, threshold)

        # If reversed, convert the frame indices back to the original order
        if reverse:
            movement_left_idx = len(left_wrist) - movement_left_idx if movement_left_idx != -1 else len(left_wrist) - 1
            movement_right_idx = len(right_wrist) - movement_right_idx if movement_right_idx != -1 else len(right_wrist) - 1

        print(f"{'Last' if reverse else 'First'} movement detected at frame {movement_left_idx} for left wrist and frame {movement_right_idx} for right wrist")
        
        return movement_left_idx, movement_right_idx
    '''
    def _find_movement_index(self, wrist_diff, threshold):
        """Helper function to find the first index exceeding the threshold."""
        movement_detected = wrist_diff > threshold
        return np.argmax(movement_detected) + 1 if np.any(movement_detected) else -1

#    def detect_movement_frames(self):
#        first = self.detect_movement_frame(threshold=0.05)
#        last = self.detect_movement_frame(threshold=0.025, sustained=True, reverse=True)
#        return first, last

    def normalize_wrist_data(self):
        normalizer = PoseNormalize(self.left_wrist, self.right_wrist, self.pose)
        self.left_wrist, self.right_wrist = normalizer.reference_normalization()
        self.left_wrist, self.right_wrist = normalizer.minmax_normalize_together()

    def print_frame_counts(self):
        counts = self.get_frame_counts()
        for key, count in counts.items():
            print(f"Number of frames in the {key} file: {count}")


def construct_boolean(pose, first, last):
    bool_L = np.zeros(len(pose))
    bool_R = np.zeros(len(pose))
    bool_L[first[0]:last[0]] = 1
    bool_R[first[1]:last[1]] = 1
    return bool_L, bool_R


def main_segmentation(base_filename, create_anim=False, save_anim_path=None, handshapes=None, handedness=None, fps = 5):
    analyzer = PoseVideoAnalyzer(base_filename)
    analyzer.print_frame_counts()
    analyzer.load_pose()
    analyzer.normalize_wrist_data()
    first, last = analyzer.detect_movement_frames()
    if create_anim:
        if save_anim_path:
            animator = WristMovementAnimator(analyzer.BASE_DIR, analyzer.left_wrist, analyzer.right_wrist, base_filename, handshapes = handshapes, handedness = handedness)
            animator.create_animation(save_path=save_anim_path, first_movement=first, last_movement=last, fps=fps)
    return construct_boolean(analyzer.pose, first, last)


if __name__ == "__main__":
    base_filename = "M20241107_6254"
    gif_path = '/home/gomer/oline/PoseTools/src/modules/demo/graphics/gifs/wrist_movement_animation.gif'
    main_segmentation(base_filename, create_anim=True, save_anim_path=gif_path)
