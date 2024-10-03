import torch
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

    def __call__(self, data:dict):
        """
        Apply selection of keypoints based on the given indexes.

        Args:
            data (dict): input data

        Returns:
            dict : transformed data
        """
        data = data[:, self.pose_indexes, ]
        return data
    
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

