import json
import os
import numpy as np
from tqdm import tqdm
import pickle


class ReadReferencePoses:
    def __init__(self, poses_dir, pdms_dir):
        """
        Initializes the class to read reference poses and PDMs from JSON files.

        Parameters:
        - poses_dir (str): The directory containing reference pose JSON files.
        - pdms_dir (str): The directory containing PDM JSON files.
        """
        self.poses_dir = poses_dir
        self.pdms_dir = pdms_dir
        self.reference_poses = {}  # To store average poses
        self.reference_pdms = {}  # To store pairwise distance matrices

    def load_poses(self):
        """
        Loads reference poses from JSON files in the poses directory.
        """
        for filename in tqdm(os.listdir(self.poses_dir), desc="Loading poses JSON files"):
            if filename.endswith(".json"):
                handshape = filename.split("_avg")[0]  # Extract the handshape name
                print(handshape)
                file_path = os.path.join(self.poses_dir, filename)
                
                with open(file_path, "r") as file:
                    data = json.load(file)
                
                for key, pose in data.items():
                    pose = np.array(pose)  # Convert pose to numpy array

                    if handshape not in self.reference_poses:
                        self.reference_poses[handshape] = []

                    self.reference_poses[handshape].append(pose)

    def load_pdms(self):
        """
        Loads pairwise distance matrices (PDMs) from JSON files in the pdms directory.
        """
        for filename in tqdm(os.listdir(self.pdms_dir), desc="Loading PDMs JSON files"):
            if filename.endswith(".json"):
                handshape = filename.split("_avg")[0]  # Extract the handshape name
                print(handshape)
                file_path = os.path.join(self.pdms_dir, filename)
                
                with open(file_path, "r") as file:
                    data = json.load(file)
                
                for key, pdm in data.items():
                    pdm = np.array(pdm)  # Convert PDM to numpy array

                    if handshape not in self.reference_pdms:
                        self.reference_pdms[handshape] = []

                    self.reference_pdms[handshape].append(pdm)

    def save_to_pickle(self, avg_pose_path, pdm_path):
        """
        Save the processed reference poses and PDMs to pickle files.

        Parameters:
        - avg_pose_path (str): Path to save the average poses.
        - pdm_path (str): Path to save the pairwise distance matrices.
        """
        for key, poses in self.reference_pdms.items():
            print(f"Key: {key}")
            for i, pose in enumerate(poses):
                print(f"  PDM {i + 1} shape: {pose.shape if hasattr(pose, 'shape') else 'Shape not available'}")
        print('\nPoses:')
        for key, poses in self.reference_poses.items():
            print(f"Key: {key}")
            for i, pose in enumerate(poses):
                print(f"  Pose {i + 1} shape: {pose.shape if hasattr(pose, 'shape') else 'Shape not available'}")

        with open(avg_pose_path, "wb") as avg_file:
            pickle.dump(self.reference_poses, avg_file)
        with open(pdm_path, "wb") as pdm_file:
            pickle.dump(self.reference_pdms, pdm_file)
        print(f"Saved average poses to {avg_pose_path}")
        print(f"Saved PDMs to {pdm_path}")


if __name__ == "__main__":
    # Directories containing reference poses and PDMs
    poses_dir = "/home/gomer/oline/PoseTools/src/modules/handshapes/utils/build_references/reference_data/finals/poses"
    pdms_dir = "/home/gomer/oline/PoseTools/src/modules/handshapes/utils/build_references/reference_data/finals/pdm"

    # Initialize the reader
    reader = ReadReferencePoses(poses_dir, pdms_dir)

    # Load poses and PDMs
    reader.load_poses()
    reader.load_pdms()

    # Paths to save pickle files
    avg_pose_pickle_path = "/home/gomer/oline/PoseTools/src/modules/handshapes/utils/build_references/references/reference_poses.pkl"
    pdm_pickle_path = "/home/gomer/oline/PoseTools/src/modules/handshapes/utils/build_references/references/reference_pdms.pkl"

    # Save the data to pickle files
    reader.save_to_pickle(avg_pose_pickle_path, pdm_pickle_path)
