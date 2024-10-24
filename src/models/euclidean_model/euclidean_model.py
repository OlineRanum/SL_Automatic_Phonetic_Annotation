
import json, os, re
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm

from PoseTools.data.parsers_and_processors.parsers import PklParser
from collections import Counter
from PoseTools.src.modules.handedness.utils.graphics import read_dict_from_txt, plot_multiple_hands_from_dict

class EvaluatePoses:
    def __init__(self, df, reference_poses, pose_dir='../../../mnt/fishbowl/gomer/oline/hamer_pkl'):
        self.pose_dir = pose_dir
        self.df = df
        self.reference_poses = reference_poses
        self.gloss_mapping = read_dict_from_txt('/home/gomer/oline/PoseTools/data/metadata/output/global_value_to_id.txt')
    

    def calculate_euclidean_distance(self, pose, reference_pose):
        """
        Calculates the Euclidean distance between a pose and a reference pose for each keypoint.
        
        Parameters:
        - pose: A numpy array of shape (21, 3), representing the pose for a frame.
        - reference_pose: A numpy array of shape (21, 3), representing the reference handshape pose.
        
        Returns:
        - The Euclidean distance between the pose and the reference pose.
        """
        return np.linalg.norm(pose - reference_pose, axis=1).mean()  # Average Euclidean distance for all keypoints

    def get_key_from_value(self, value):
        """
        Returns the key in the gloss_mapping dictionary that corresponds to the given value.
        
        Parameters:
        - gloss_mapping: A dictionary where keys are integers and values are handshapes.
        - value: The value (handshape) for which the key is needed.
        
        Returns:
        - The key corresponding to the given value, or None if the value is not found.
        """
        
        for key, val in self.gloss_mapping.items():
            
            if val == value:
                return key
        return None  # Return None if the value is not found
    
    def get_value_from_key(self, value):
        """
        Returns the key in the gloss_mapping dictionary that corresponds to the given value.
        
        Parameters:
        - gloss_mapping: A dictionary where keys are integers and values are handshapes.
        - value: The value (handshape) for which the key is needed.
        
        Returns:
        - The key corresponding to the given value, or None if the value is not found.
        """
        
        for key, val in self.gloss_mapping.items():
            
            if key == value:
                return val
        return None  # Return None if the value is not found
    
    def distance_per_frame(self, pose_data):
        """
        Evaluates each frame by calculating the Euclidean distance to the reference handshapes,
        and determines the median handshape across all frames.
        """
        num_frames = pose_data.shape[0]
        closest_handshapes = []  # List to store the closest handshape per frame
        distances = [] 
        for frame_idx in range(num_frames):
            pose = pose_data[frame_idx]  # Pose at the current frame (21, 3)
            
            # Calculate the Euclidean distance to both handshapes
            distances = []
            keys = []
            for key, reference_pose in self.reference_poses.items():
                distance = self.calculate_euclidean_distance(pose, reference_pose)
                keys.append(key)
                distances.append(distance)

            closest_handshape = self.gloss_mapping[int(keys[np.argmin(np.array(distances))])]
            
            closest_handshapes.append(closest_handshape)
        
        
        handshape_counter = Counter(closest_handshapes)
        try:
            median_handshape = handshape_counter.most_common(1)[0][0]  # Get the most frequent handshape
        except IndexError:
            print('!!!!!!!!!')
            median_handshape = 'h1'
        return median_handshape
    
    def euclidean_distance_per_frame(self, pose_data, reference_pose):
        """
        Evaluates each frame by calculating the Euclidean distance to the reference handshapes (h1 and h2),
        and determines the median handshape across all frames.
        
        Parameters:
        - pose_data: A numpy array of shape (num_frames, 21, 3), representing the pose data for each frame.
        - reference_pose_h1: The reference pose for handshape h1 (21, 3).
        - reference_pose_h2: The reference pose for handshape h2 (21, 3).
        - video_id: The video ID for which the evaluation is performed.
        
        Prints the median handshape for the entire video.
        """
        num_frames = pose_data.shape[0]
        distances = []  # List to store the closest handshape per frame

        for frame_idx in range(num_frames):
            pose = pose_data[frame_idx]  # Pose at the current frame (21, 3)
            
            # Calculate the Euclidean distance to both handshapes
            dist = self.calculate_euclidean_distance(pose, reference_pose)
            distances.append(dist)
        
        mean_distance = np.mean(np.array(distances))  # Get the most frequent handshape

        return mean_distance

    def get_pose(self, video_id, reference_pose):
        if os.path.exists(video_id):
            parser = PklParser(input_path=video_id)
            pose_data, _ = parser.read_pkl()  # pose_data shape is (num_frames, 21, 3)
            return self.evaluate_per_frame(pose_data, reference_pose, None, video_id)
        else: 
            return float('inf')
    
    def get_distance(self, video_id, reference_pose):
        if os.path.exists(video_id):
            parser = PklParser(input_path=video_id)
            pose_data, _ = parser.read_pkl()  # pose_data shape is (num_frames, 21, 3)
            return self.euclidean_distance_per_frame(pose_data, reference_pose)
        else: 
            return float('inf')
        
    def evaluate_poses(self):
        """
        Evaluates poses for each row in the DataFrame by checking if pose files exist for -L and -R,
        and counts the number of existing files.
        """

        output_file = 'PoseTools/results/euclidean_sb.txt'
        self.gloss_mapping = read_dict_from_txt('/home/gomer/oline/PoseTools/data/metadata/output/global_value_to_id.txt')
        
        
        # Open the output file in write mode
        with open(output_file, 'w') as f_out:
            for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing Poses"):
       
                
                # Get the video_id from the row
                video_id = row['video_id']
                filename = row['filename']
                handedness = row['Sign Type']
                hand_L = self.get_distance(filename, view='-L')
                if hand_L is not None:
                    f_out.write(f"{filename}, {handedness}, {video_id}-L, {hand_L}\n")
                
                hand_R = self.get_distance(filename, view='-R')
                if hand_R is not None:
                    f_out.write(f"{filename}, {handedness}, {video_id}-R, {hand_R}\n")


if __name__ == "__main__":
    df = pd.read_csv('/home/gomer/oline/PoseTools/src/modules/handedness/euclidean_model/ground_truth_handshape.csv', delimiter=',', na_values='Unknown')
    # Load the object from the file
    df = df.rename(columns={'gloss': 'video_id', 'handedness': 'Sign Type', 'strong_hand': 'Handshape', 'weak_hand': 'Nondominant Handshape'})
    with open('/home/gomer/oline/PoseTools/src/modules/handedness/euclidean_model/reference_poses.pkl', 'rb') as file:
        reference_poses = pickle.load(file)
    
    # Create an instance of the class
    evaluator = EvaluatePoses(df, reference_poses)

    # Call the evaluate_poses method to print video IDs
    evaluator.evaluate_poses()


