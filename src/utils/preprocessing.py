import torch
import numpy as np
class PoseSelect:
    """
    Select the given index keypoints from all keypoints. 
    
    Args:
        preset (str | None, optional): can be used to specify existing presets - `mediapipe_holistic_minimal_27` or `mediapipe_holistic_top_body_59`
        If None, then the `pose_indexes` argument indexes will be used to select. Default: ``None``
        
        pose_indexes: List of indexes to select.
    """
    # fmt: off
    KEYPOINT_PRESETS = {
        "mediapipe_holistic_minimal_27": [ 0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53, 54, 58, 59, 62, 63, 66, 67, 70, 71, 74],
        "mediapipe_holistic_top_body_59": [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 23, 24, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74]
    }
    # fmt: on
    
    def __init__(self, preset=None, pose_indexes: list=[]):
        if preset:
            self.pose_indexes = self.KEYPOINT_PRESETS[preset]
        elif pose_indexes:
            self.pose_indexes = pose_indexes
        else:
            raise ValueError("Either pass `pose_indexes` to select or `preset` name")

    def __call__(self, data):
        """
        Apply selection of keypoints based on the given indexes.

        Args:
            data (dict): input data

        Returns:
            dict : transformed data
        """
        data = data.squeeze(1)
        data = data[:, self.pose_indexes, :]
        return data
    
    def get_subset(self, data):
        return data[:, self.pose_indexes, :]
    
    def clean_keypoints(self, pose):
        """
        Clean keypoints by removing surplus data.

        :param pose: The pose data to clean.
        :return: Cleaned pose data.
        """
        try:
            pose = pose[:, :, :543, :].squeeze(1)
        except IndexError:
            pose = pose[:,:, :543].squeeze(1)
        return pose

    def get_keypoints_face(self, pose):
        """
        Extract keypoints related to the face.

        :param pose: The pose data from which to extract face keypoints.
        :return: Keypoints related to the face.
        """
        
        return pose[:, 33:-42, :]

    def get_keypoints_pose(self, pose):
        """
        Extract keypoints related to the pose.

        :param pose: The pose data from which to extract pose keypoints.
        :return: Keypoints related to the pose.
        """
        try:
            pose = pose[:, :33, :]
        except IndexError:
            pose = pose[:, :33]
        return pose

    def get_keypoints_hands(self, pose):
        """
        Extract keypoints related to the hands.

        :param pose: The pose data from which to extract hand keypoints.
        :return: Keypoints related to the hands.
        """
        try:
            pose = pose[:, -42:, :]
        except IndexError:
            pose = pose[:, :-42]
        return pose
    
    def get_left_wrist(self, pose):
        return pose[:, 15, :]
    
    def get_right_wrist(self, pose):
        return pose[:, 16, :]


    def get_keypoints_pose_hands_face(self, pose):
        """
        Extract keypoints related to the pose and hands and return them as a single array.

        :param pose: The pose data from which to extract pose and hand keypoints.
        :return: Combined keypoints related to the pose and hands.
        """
        import numpy as np
        pose_keypoints = self.get_keypoints_pose(pose)  # Extract pose keypoints
        hand_keypoints = self.get_keypoints_hands(pose)  # Extract hand keypoints
        face_keypoints = self.get_keypoints_face(pose)  # Extract face keypoints
        combined_keypoints = np.concatenate((pose_keypoints, hand_keypoints, face_keypoints), axis=1)  # Combine them along the appropriate axis
        return combined_keypoints

    def get_keypoints_pose_and_hands(self, pose):
        """
        Extract keypoints related to the pose and hands and return them as a single array.

        :param pose: The pose data from which to extract pose and hand keypoints.
        :return: Combined keypoints related to the pose and hands.
        """
        import numpy as np
        pose_keypoints = self.get_keypoints_pose(pose)  # Extract pose keypoints
        hand_keypoints = self.get_keypoints_hands(pose)  # Extract hand keypoints
        combined_keypoints = np.concatenate((pose_keypoints, hand_keypoints), axis=1)  # Combine them along the appropriate axis
        return combined_keypoints

# PoseTools/src/utils/preprocessing.py

import numpy as np

class PoseNormalize:
    def __init__(self, left_wrist, right_wrist, pose):
        """
        Initialize with left and right wrist positions and pose landmarks.

        Parameters:
            left_wrist (np.ndarray): Left wrist positions, shape [T_frames, 2].
            right_wrist (np.ndarray): Right wrist positions, shape [T_frames, 2].
            pose (np.ndarray): Pose landmarks, shape [T_frames, N_landmarks, 2].
        """
        # Preserve raw wrist positions
        self.pose = pose.copy()
        if left_wrist is None or right_wrist is None:
            pass
        else:
            self.left_wrist_raw = left_wrist.copy()    # Shape: [T_frames, 2]
            self.right_wrist_raw = right_wrist.copy()  # Shape: [T_frames, 2]
                        # Shape: [T_frames, N_landmarks, 2]
            self.left_wrist = left_wrist.copy()    # Shape: [T_frames, 2]
            self.right_wrist = right_wrist.copy()  # Shape: [T_frames, 2]
        
        # Extract relevant landmarks
        # Assuming landmark indices:
        # 0: Nose
        # 11: Left Shoulder
        # 12: Right Shoulder
        # 23: Left Hip
        # 24: Right Hip
 
    def fullpose_normalization(self, pose = None):
        if pose is not None:
            self.pose = pose
        self.L_shoulder = self.pose[:, 11, :]  # Left shoulder, shape: [T_frames, 2]
        self.R_shoulder = self.pose[:, 12, :]  # Right shoulder, shape: [T_frames, 2]
        self.nose = self.pose[:, 0, :]          # Nose, shape: [T_frames, 2]
        self.L_hip = self.pose[:, 23, :]        # Left hip, shape: [T_frames, 2]
        self.R_hip = self.pose[:, 24, :]        # Right hip, shape: [T_frames, 2]
        self.waist = (self.L_hip + self.R_hip) / 2  # Waist position, shape: [T_frames, 2]
   
         # Ensure pose has 3 coordinates
        T, N, _ = self.pose.shape

        # Initialize normalized pose array
        normalized_pose = np.zeros_like(self.pose)

        # Small epsilon to prevent division by zero
        epsilon = 1e-8

        # Iterate over each frame for per-frame normalization
        for t in range(T):

            # --- X-axis Normalization ---
            L_sh_x = self.L_shoulder[t, 0]
            R_sh_x = self.R_shoulder[t, 0]
            x_diff = R_sh_x - L_sh_x
            if np.abs(x_diff) < epsilon:
                x_diff = epsilon  # Prevent division by zero
            x_norm = (self.pose[t, :, 0] - L_sh_x) / x_diff  # Shape: [N]
            #print('x ',np.max(x_norm), np.min(x_norm))
            # --- Y-axis Normalization ---
            shoulders_y = (self.L_shoulder[t, 1] + self.R_shoulder[t, 1]) / 2
            waist_y = self.waist[t, 1]
            y_diff = shoulders_y - waist_y
            if np.abs(y_diff) < epsilon:
                y_diff = epsilon
            y_norm = (self.pose[t, :, 1] - waist_y) / y_diff  # Shape: [N]
            #print('y', np.max(y_norm), np.min(y_norm))
            # --- Z-axis Normalization ---
            nose_z = self.nose[t, 2]
            waist_z = self.waist[t, 2]
            z_diff = nose_z - waist_z
            if np.abs(z_diff) < epsilon:
                z_diff = epsilon
            z_norm = (self.pose[t, :, 2] - waist_z) / z_diff  # Shape: [N]
            #print('z', np.max(z_norm), np.min(z_norm))
            # Stack normalized coordinates for this frame
            normalized_pose[t, :, 0] = x_norm
            normalized_pose[t, :, 1] = y_norm
            normalized_pose[t, :, 2] = z_norm

        # Optional: Clip values to [0, 1] if desired
        # normalized_pose = np.clip(normalized_pose, 0, 1)

        return normalized_pose

    def reference_normalization(self, wrist_left, wrist_right):
        """
        Normalize wrist positions relative to the initial frame.
        Subtract the position at the first frame and take the absolute value.
        
        Returns:
            tuple: Normalized left and right wrist magnitudes.
        """
        self.left_wrist = wrist_left.copy()
        self.right_wrist = wrist_right.copy()
        self.left_wrist = np.linalg.norm(self.left_wrist, axis=1)
        self.left_wrist -= self.left_wrist[0]
        self.left_wrist = np.abs(self.left_wrist)
        
        self.right_wrist = np.linalg.norm(self.right_wrist, axis=1)
        self.right_wrist -= self.right_wrist[0]
        self.right_wrist = np.abs(self.right_wrist)
        
        return self.left_wrist, self.right_wrist

    def minmax_normalize_together(self, wrist_left, wrist_right):
        """
        Min-max normalize both wrist arrays together based on their combined range.
        
        Returns:
            tuple: Min-max normalized left and right wrist magnitudes.
        """
        if wrist_left is not None:
            self.left_wrist = wrist_left.copy()
        if wrist_right is not None:
            self.right_wrist = wrist_right.copy()
        combined = np.concatenate((self.left_wrist, self.right_wrist))
        min_val = combined.min()
        max_val = combined.max()
        
        if max_val != min_val:
            self.left_wrist = (self.left_wrist - min_val) / (max_val - min_val)
            self.right_wrist = (self.right_wrist - min_val) / (max_val - min_val)
        else:
            # If all values are the same, normalization is not needed
            pass
        
        return self.left_wrist, self.right_wrist
    
    '''
    def reference_normalization(self):
        """
        Normalize wrist positions relative to the initial frame.
        Subtract the position at the first frame and take the absolute value.

        Returns:
            tuple: Normalized left and right wrist magnitudes (1D arrays).
        """
        # Calculate magnitudes (Euclidean norms)
        left_magnitude = np.linalg.norm(self.left_wrist_raw, axis=1)
        right_magnitude = np.linalg.norm(self.right_wrist_raw, axis=1)
        
        # Reference normalization: subtract the initial magnitude and take absolute value
        l0 = left_magnitude[0]
        left_magnitude -= l0
        left_magnitude = np.abs(left_magnitude)
        left_magnitude += l0

        r0 = right_magnitude[0]
        right_magnitude -= r0
        right_magnitude = np.abs(right_magnitude)
        right_magnitude += r0
                
        # Store normalized magnitudes
        self.left_wrist_normalized_ref = left_magnitude
        self.right_wrist_normalized_ref = right_magnitude
        
        return self.left_wrist_normalized_ref, self.right_wrist_normalized_ref
    
    def minmax_normalize_together(self):
        """
        Min-max normalize both wrist arrays together based on their combined range.

        Returns:
            tuple: Min-max normalized left and right wrist magnitudes (1D arrays).
        """
        # Calculate magnitudes
        left_magnitude = np.linalg.norm(self.left_wrist_raw, axis=1)
        right_magnitude = np.linalg.norm(self.right_wrist_raw, axis=1)
        
        # Combine for joint min-max
        combined = np.concatenate((left_magnitude, right_magnitude))
        min_val = combined.min()
        max_val = combined.max()
        
        if max_val != min_val:
            left_normalized = (left_magnitude - min_val) / (max_val - min_val)
            right_normalized = (right_magnitude - min_val) / (max_val - min_val)
        else:
            # If all values are the same, set to zero
            left_normalized = np.zeros_like(left_magnitude)
            right_normalized = np.zeros_like(right_magnitude)
        
        # Store normalized magnitudes
        self.left_wrist_normalized_minmax = left_normalized
        self.right_wrist_normalized_minmax = right_normalized
        
        return self.left_wrist_normalized_minmax, self.right_wrist_normalized_minmax
    '''
    def normalize_by_landmarks(self):
        """
        Normalize wrist positions so that:
            - X-axis: 0 corresponds to the right shoulder's mean X position, 1 to the left shoulder's mean X position.
            - Y-axis: 0 corresponds to the waist's mean Y position, 1 to the nose's mean Y position.

        This maps the wrist positions within a [0, 1] range based on these fixed mean landmarks.

        Returns:
            tuple: Normalized left and right wrist positions, each of shape [T_frames, 2].
        """
        # Compute mean positions across all frames
        
        mean_L_shoulder_x = np.mean(self.L_shoulder[:, 0])
        mean_R_shoulder_x = np.mean(self.R_shoulder[:, 0])
        mean_waist_y = np.mean(self.waist[:, 1])
        mean_nose_y = np.mean(self.nose[:, 1])
        
        # Debugging: Print the mean positions
        #print(f"Mean Right Shoulder X: {mean_R_shoulder_x}")
        #print(f"Mean Left Shoulder X: {mean_L_shoulder_x}")
        #print(f"Mean Waist Y: {mean_waist_y}")
        #print(f"Mean Nose Y: {mean_nose_y}")
        
        # Compute ranges for normalization
        x_range = mean_L_shoulder_x - mean_R_shoulder_x  # Should be >0 if left shoulder is to the right
        y_range = mean_nose_y - mean_waist_y             # Should be >0 if nose is above waist
        
        # Handle division by zero
        if x_range == 0:
            print("Warning: Left and Right shoulders have the same mean X position. Setting normalized X to 0.5.")
            normalized_left_wrist_x = np.full(self.left_wrist_raw.shape[0], 0.5)
            normalized_right_wrist_x = np.full(self.right_wrist_raw.shape[0], 0.5)
        else:
            # Normalize X positions
            normalized_left_wrist_x = (self.left_wrist_raw[:, 0] - mean_R_shoulder_x) / x_range
            normalized_right_wrist_x = (self.right_wrist_raw[:, 0] - mean_R_shoulder_x) / x_range
            
            # Clip to [0, 1]
            normalized_left_wrist_x = np.clip(normalized_left_wrist_x, 0, 1)
            normalized_right_wrist_x = np.clip(normalized_right_wrist_x, 0, 1)
        
        if y_range == 0:
            print("Warning: Nose and Waist have the same mean Y position. Setting normalized Y to 0.5.")
            normalized_left_wrist_y = np.full(self.left_wrist_raw.shape[0], 0.5)
            normalized_right_wrist_y = np.full(self.right_wrist_raw.shape[0], 0.5)
        else:
            # Normalize Y positions
            normalized_left_wrist_y = (self.left_wrist_raw[:, 1] - mean_waist_y) / y_range
            normalized_right_wrist_y = (self.right_wrist_raw[:, 1] - mean_waist_y) / y_range
            
            # Clip to [0, 1]
            normalized_left_wrist_y = np.clip(normalized_left_wrist_y, 0, 1)
            normalized_right_wrist_y = np.clip(normalized_right_wrist_y, 0, 1)
        
        # Combine the normalized X and Y coordinates
        self.left_wrist_normalized_landmarks = np.stack((normalized_left_wrist_x, normalized_left_wrist_y), axis=1)
        self.right_wrist_normalized_landmarks = np.stack((normalized_right_wrist_x, normalized_right_wrist_y), axis=1)
        
        print(f"Max data L, R: {np.max(self.left_wrist_normalized_landmarks[:, 0])}, {np.max(self.left_wrist_normalized_landmarks[:, 0])}")
        print(f"Min data L, R: {np.min(self.right_wrist_normalized_landmarks[:, 0])}, {np.min(self.right_wrist_normalized_landmarks[:, 0])}")
        self.left_wrist_raw = self.left_wrist_normalized_landmarks
        self.right_wrist_raw = self.right_wrist_normalized_landmarks
        return self.left_wrist_normalized_landmarks, self.right_wrist_normalized_landmarks

class CenterAndScaleNormalize:
    """
    Centers and scales the keypoints based on the referent points given.

    Args:
        reference_points_preset (str | None, optional): can be used to specify existing presets - `mediapipe_holistic_minimal_27` or `mediapipe_holistic_top_body_59`
        reference_point_indexes (list): shape(p1, p2); point indexes to use if preset is not given then
        scale_factor (int): scaling factor. Default: 1
        frame_level (bool): Whether to center and normalize at frame level or clip level. Default: ``False``
    """
    REFERENCE_PRESETS = {
        "shoulder_mediapipe_holistic_minimal_27": [3, 4],
        "shoulder_mediapipe_holistic_top_body_59": [11, 12],
    }

    def __init__(
        self,
        reference_points_preset=None,
        reference_point_indexes=[],
        scale_factor=1,
        frame_level=False,
    ):

        if reference_points_preset:
            self.reference_point_indexes = CenterAndScaleNormalize.REFERENCE_PRESETS[
                reference_points_preset
            ]
        elif reference_point_indexes:
            self.reference_point_indexes = reference_point_indexes
        else:
            raise ValueError(
                "Mention the joint with respect to which the scaling & centering must be done"
            )
        self.scale_factor = scale_factor
        self.frame_level = frame_level

    def __call__(self, data):
        """
        Applies centering and scaling transformation to the given data.

        Args:
            data (dict): input data

        Returns:
            dict: data after centering normalization
        """
        try:
            x = data["frames"]
        except IndexError:
            x = data
        C, T, V = x.shape
        x = x.permute(1, 2, 0) #CTV->TVC

        if self.frame_level:
            for ind in range(x.shape[0]):
                center, scale = self.calc_center_and_scale_for_one_skeleton(x[ind])
                x[ind] -= center
                x[ind] *= scale
        else:
            center, scale = self.calc_center_and_scale(x)
            x = x - center
            x = x * scale

        data = x.permute(2, 0, 1) #TVC->CTV
        return data

    def calc_center_and_scale_for_one_skeleton(self, x):
        """
        Calculates the center and scale values for one skeleton.

        Args:
            x (torch.Tensor): Spatial keypoints at a timestep

        Returns:
            [float, float]: center and scale value to normalize for the skeleton
        """
        ind1, ind2 = self.reference_point_indexes
        point1, point2 = x[ind1], x[ind2]
        center = (point1 + point2) / 2
        dist = torch.sqrt(((point1 - point2) ** 2).sum(-1))
        scale = self.scale_factor / dist
        if torch.isinf(scale).any():
            return 0, 1  # Do not normalize
        return center, scale

    def calc_center_and_scale(self, x):
        """
        Calculates the center and scale value based on the sequence of skeletons.

        Args:
            x (torch.Tensor): all keypoints for the video clip.

        Returns:
            [float, float]: center and scale value to normalize
        """
        transposed_x = x.permute(1, 0, 2) # TVC -> VTC
        ind1, ind2 = self.reference_point_indexes
        points1 = transposed_x[ind1]
        points2 = transposed_x[ind2]

        points1 = points1.reshape(-1, points1.shape[-1])
        points2 = points2.reshape(-1, points2.shape[-1])

        center = torch.mean((points1 + points2) / 2, dim=0)
        mean_dist = torch.mean(torch.sqrt(((points1 - points2) ** 2).sum(-1)))
        scale = self.scale_factor / mean_dist
        if torch.isinf(scale).any():
            return 0, 1  # Do not normalize

        return center, scale

