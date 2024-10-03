
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