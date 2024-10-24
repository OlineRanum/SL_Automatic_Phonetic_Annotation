
import json, os, re
import pandas as pd
import pickle
import numpy as np

from PoseTools.data.parsers_and_processors.parsers import PklParser
from collections import Counter
from PoseTools.src.modules.handedness.utils.graphics import read_dict_from_txt, plot_multiple_hands_from_dict

class EvaluatePoses:
    def __init__(self, df, reference_poses):
        self.pose_dir = '../../../../mnt/fishbowl/gomer/oline/hamer_large_pkl'
        self.df = df
        self.reference_poses = reference_poses

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

    def evaluate_per_frame(self, pose_data, reference_pose_h1, reference_pose_h2, video_id, method = '1'):
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
        closest_handshapes = []  # List to store the closest handshape per frame
        distance_h1 = []  # List to store the distances for each frame
        distance_h2 = []
        for frame_idx in range(num_frames):
            pose = pose_data[frame_idx]  # Pose at the current frame (21, 3)
            
            # Calculate the Euclidean distance to both handshapes
            if reference_pose_h1 is not None:
                dist_h1 = self.calculate_euclidean_distance(pose, reference_pose_h1)
                distance_h1.append(dist_h1)
            else:
                dist_h1 = float('inf')
            if reference_pose_h2 is not None:
                dist_h2 = self.calculate_euclidean_distance(pose, reference_pose_h2)
                distance_h2.append(dist_h2)
            else:
                dist_h2 = float('inf')

            # Determine which handshape is closer
            if dist_h1 < dist_h2:
                closest_handshape = "h1"

            elif dist_h2 < dist_h1:
                closest_handshape = "h2"
            else:
                
                continue  # Skip if both distances are equal
            
            closest_handshapes.append(closest_handshape)
        #print('h1 =', np.mean(np.array(distance_h1)))
        #print('h2 =',np.mean(np.array(distance_h2)))    
        
        # Find the median handshape by determining the most common one
        #print(closest_handshapes)
        handshape_counter = Counter(closest_handshapes)
        try:
            median_handshape = handshape_counter.most_common(1)[0][0]  # Get the most frequent handshape
        except IndexError:
            print('!!!!!!!!!')
            median_handshape = 'h1'
        # Print the result for the entire video
        if method == '1':
            if median_handshape == 'h1':
                return np.mean(np.array(distance_h1))
            elif median_handshape == 'h2':
                return np.mean(np.array(distance_h2))
            else:
                return None
        elif method == '2a':
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
            return self.euclidean_distance_per_frame(pose_data, reference_pose, None, video_id)
        else: 
            return float('inf')

    
    def evaluate_poses(self):
        """
        Evaluates poses for each row in the DataFrame by checking if pose files exist for -L and -R,
        and counts the number of existing files.
        """

        output_file = 'PoseTools/results/euclidean.txt'
        self.gloss_mapping = read_dict_from_txt('/home/gomer/oline/PoseTools/data/metadata/output/global_value_to_id.txt')
        
        
        # Open the output file in write mode
        with open(output_file, 'w') as f_out:
            for _, row in self.df.iterrows():
                
                # Get the video_id from the row
                video_id = row['video_id']
                filename = row['filename']
                handedness = row['Sign Type']
                h1 = row['Handshape']
                h2 = row['Nondominant Handshape']
                video_id_L = os.path.join(self.pose_dir, video_id + '-L.pkl')
                video_id_R = os.path.join(self.pose_dir, video_id + '-R.pkl')
                try:
                    if handedness == '1':
                        h1_key = str(self.get_key_from_value(h1))

                        reference_pose_h1 = self.reference_poses[h1_key]
                        
                        
                        distance_L = self.get_distance(video_id_L, reference_pose_h1)
                        distance_R = self.get_distance(video_id_R, reference_pose_h1)
                                
                        if distance_L < distance_R:
                            f_out.write(f"{filename}, 1, {video_id}-L, {h1_key}, {h1}\n")
                        elif distance_R < distance_L:
                            f_out.write(f"{filename}, 1, {video_id}-R, {h1_key}, {h1}\n")
                        
                            
                    elif (handedness == '2s') or (handedness == '2a' and h1 == h2):
                        h1_key = str(self.get_key_from_value(h1))
                        if os.path.exists(video_id_L):
                            f_out.write(f"{filename}, 2s, {video_id}-L, {h1_key}, {h1}\n")
                        if os.path.exists(video_id_R):
                            f_out.write(f"{filename}, 2s, {video_id}-R, {h1_key}, {h1}\n")
                    
                    elif handedness == '2a' and h1 != h2:
                        
                        h1_key = self.get_key_from_value(h1)
                        h2_key = self.get_key_from_value(h2)
                        if h2_key is not None: h2_key = str(h2_key)
                        if h1_key is not None: h1_key = str(h1_key)
                        
                        # Initialize reference poses
                        reference_pose_h1, reference_pose_h2 = None, None
                        
                        # Check if h1_key exists and retrieve the reference pose for h1
                        if h1_key is not None and h1_key in self.reference_poses:
                            reference_pose_h1 = self.reference_poses[h1_key]
                        else:
                            print(f"Reference handshape not found for h1 ({h1}) in {video_id}")
                        
                        # Check if h2_key exists and retrieve the reference pose for h2
                        if h2_key is not None and h2_key in self.reference_poses:
                            reference_pose_h2 = self.reference_poses[h2_key]

                        else:
                            print(f"Reference handshape not found for h2 ({h2}) in {video_id}")
                        
                        # Skip row if neither h1 nor h2 is found
                        if reference_pose_h1 is None and reference_pose_h2 is None:
                            print(f"Skipping {video_id}: No valid reference handshapes found.")
                            continue

                        # Construct paths for -L and -R versions of the video ID
                        video_id_base = video_id  # Remove the last two characters from video_id
                        
                        video_id_L = os.path.join(self.pose_dir, video_id + '-L.pkl')
                        video_id_R = os.path.join(self.pose_dir, video_id + '-R.pkl')

                        hand_L = None
                        hand_R = None
                        
                        # Check if the -L file exists and evaluate (for h1 or h2 if available)
                        if os.path.exists(video_id_L):
                            parser = PklParser(input_path=video_id_L)
                            pose_data, _ = parser.read_pkl()  # pose_data shape is (num_frames, 21, 3)
                            hand_L = self.evaluate_per_frame(pose_data, reference_pose_h1, reference_pose_h2, video_id_L, method = '2a')
                            hand_L_ = hand_L
                            if hand_L == 'h1': hand_L = h1
                            elif hand_L == 'h2': hand_L = h2
                            else: hand_L = None
                            # Write video_ID, L, hand_L to the output file
                            
                            
                            if hand_L is not None:
                                h_key = str(self.get_key_from_value(hand_L))
                                f_out.write(f"{filename}, 2a, {video_id}-L, {h_key}, {hand_L}\n")
                                                
                        # Check if the -R file exists and evaluate (for h1 or h2 if available)
                        if os.path.exists(video_id_R):
                            parser = PklParser(input_path=video_id_R)
                            pose_data, _ = parser.read_pkl()  # pose_data shape is (num_frames, 21, 3)
                            hand_R = self.evaluate_per_frame(pose_data, reference_pose_h1, reference_pose_h2, video_id_R, method = '2a')
                            
                            if hand_R == 'h1': hand_R = h1
                            elif hand_R == 'h2': hand_R = h2
                            else: hand_R = None
                            # Write video_ID, R, hand_R to the output file
                            if hand_R is not None:
                                h_key = str(self.get_key_from_value(hand_R))
                                f_out.write(f"{filename}, 2a, {video_id}-R, {h_key} {hand_R}\n")
                except Exception as e:  
                    continue

if __name__ == "__main__":
    df = pd.read_csv('PoseTools/src/modules/handedness/euclidean_model/ground_truth_handshape.csv', delimiter=',', na_values='Unknown')
    # Load the object from the file
    df = df.rename(columns={'gloss': 'video_id', 'handedness': 'Sign Type', 'strong_hand': 'Handshape', 'weak_hand': 'Nondominant Handshape'})
    with open('/home/gomer/oline/PoseTools/src/modules/handedness/model/reference_poses.pkl', 'rb') as file:
        reference_poses = pickle.load(file)
    
    # Create an instance of the class
    evaluator = EvaluatePoses(df, reference_poses)

    # Call the evaluate_poses method to print video IDs
    evaluator.evaluate_poses()


