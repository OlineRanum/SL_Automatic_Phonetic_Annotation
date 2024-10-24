
import json, os, re
import pandas as pd
import pickle
import numpy as np

from PoseTools.data.parsers_and_processors.parsers import PklParser
from collections import Counter
from PoseTools.src.modules.handedness.utils.graphics import read_dict_from_txt, plot_multiple_hands_from_dict


def json_to_dataframe(json_file):
    # Load the JSON data from the file
    with open(json_file, 'r') as f:
        gloss_data = json.load(f)

    # Create a list to hold the instance data
    data = []

    # Iterate over gloss entries
    for gloss_entry in gloss_data:
        gloss = gloss_entry['gloss']  # Extract the gloss
        instances = gloss_entry['instances']  # Extract the instances list

        # Iterate over each instance and append the gloss to the instance details
        for instance in instances:
            instance['gloss'] = gloss  # Add gloss to the instance
            data.append(instance)  # Add the instance data to the list

    # Convert the list of instance dictionaries into a pandas DataFrame
    df = pd.DataFrame(data)
    print(len(df))
    
    return df


def split_dataframe_by_handedness(df):
    # Split the DataFrame based on the 'Handedness' column
    df_1 = df[(df['Sign Type'] == '1')]
    df_2s = df[(df['Sign Type'] == '2s')]
    df_2a = df[df['Sign Type'] == '2a']
    
    return df_1, df_2s, df_2a


def splitLR_dataframe(df):
    """
    Adds a column indicating whether the video_id refers to the left (-L) or right (-R) hand,
    and splits the dataframe into two separate dataframes based on the tail of the video_id.
    
    Parameters:
    - df: The original dataframe that contains a 'video_id' column.
    
    Returns:
    - df_L: A dataframe with only '-L' entries in the 'video_id'.
    - df_R: A dataframe with only '-R' entries in the 'video_id'.
    """
    # Step 1: Add a new column 'hand' which is either 'L' or 'R' based on the tail of 'video_id'
    df['hand'] = df['video_id'].apply(lambda x: 'L' if x.endswith('-L') else 'R')

    # Step 2: Split the dataframe into two dataframes based on the 'hand' column
    df_L = df[df['hand'] == 'L'].copy().reset_index(drop=True)
    df_R = df[df['hand'] == 'R'].copy().reset_index(drop=True)

    return df_L, df_R


class BuildReferencePose:
    def __init__(self):
        self.pose_dir = 'PoseTools/data/datasets/hamer_1_2s_2a/normalized'
        self.handshape_dict = {}  # Dictionary to store handshape classes and their corresponding poses
        self.reference_poses = {}  # Dictionary to store the final reference poses
        self.gloss_mapping = read_dict_from_txt('/home/gomer/oline/PoseTools/data/metadata/output/global_value_to_id.txt')
    
    def build_reference_pose(self, df):
        # Loop through each row and group poses by 'gloss' (handshape class)
        for index, row in df.iterrows():
            video_id = row['video_id']
            handedness = video_id.split('-')[-1]
            filepath = os.path.join(self.pose_dir, f"{video_id}.pkl")
            try:
                with open(filepath, 'rb') as f:
                    data_dict = pickle.load(f)
            except FileNotFoundError:
                continue    
            # Extract keypoints data
            data = data_dict.get('keypoints', None)
            
            # Average across time steps (axis=0), resulting in [n_nodes, spatial_dims]
            avg_pose = data.mean(axis=0)
            
            # Get the handshape class (gloss)
            gloss = row['gloss']
            
            # Add the average pose to the dictionary for this class
            if gloss not in self.handshape_dict:
                self.handshape_dict[gloss] = []
            self.handshape_dict[gloss].append(avg_pose)
        
        # Calculate the final average pose for each class
        total_poses = 0
        for gloss, poses in self.handshape_dict.items():
            # Convert the list of poses to a NumPy array
            poses = np.array(poses)
            
            # Average across all instances for the current class (axis=0)
            avg_class_pose = poses.mean(axis=0)
            
            # Print the number of instances used for this class
            #print(f"Gloss: {gloss}, Number of instances: {len(poses)}")
            self.reference_poses[gloss] = avg_class_pose
            #plot_hamer_hand_3d(avg_class_pose, gloss)
            
            total_poses += len(poses)
    
        print(f"Total number of poses: {total_poses}")
        return self.reference_poses
    




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
    
    def get_distance(self, video_id, reference_pose):
        if os.path.exists(video_id):
            parser = PklParser(input_path=video_id)
            pose_data, _ = parser.read_pkl()  # pose_data shape is (num_frames, 21, 3)
            return self.evaluate_per_frame(pose_data, reference_pose, None, video_id)
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
                handedness = row['Sign Type']
                h1 = row['Handshape']
                h2 = row['Nondominant Handshape']
                video_id_L = os.path.join(self.pose_dir, video_id + '-L.pkl')
                video_id_R = os.path.join(self.pose_dir, video_id + '-R.pkl')

                if handedness == '1':
                    h1_key = str(self.get_key_from_value(h1))
                    reference_pose_h1 = self.reference_poses[h1_key]
                    
                    distance_L = self.get_distance(video_id_L, reference_pose_h1)
                    distance_R = self.get_distance(video_id_R, reference_pose_h1)
                            
                    if distance_L < distance_R:
                        f_out.write(f"1, {video_id}-L, {h1_key}, {h1}\n")
                    elif distance_R < distance_L:
                        f_out.write(f"1, {video_id}-R, {h1_key}, {h1}\n")
                    
                        
                elif (handedness == '2s') or (handedness == '2a' and h1 == h2):
                    h1_key = str(self.get_key_from_value(h1))
                    if os.path.exists(video_id_L):
                        f_out.write(f"2s, {video_id}-L, {h1_key}, {h1}\n")
                    if os.path.exists(video_id_R):
                        f_out.write(f"2s, {video_id}-R, {h1_key}, {h1}\n")
                
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
                            f_out.write(f"2a, {video_id}-L, {h_key}, {hand_L}\n")
                                            
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
                            f_out.write(f"2a, {video_id}-R, {h_key} {hand_R}\n")
                    
                        


def search_reference_poses(df, reference_directory):
    """
    Searches for reference pose files (with +R or +L suffix) in the reference directory.
    If such files exist, adds the updated video_id with the suffix and the row to a new DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing the video_id column without suffixes
        reference_directory (str): Path to the directory containing reference pose files
        
    Returns:
        pd.DataFrame: A new DataFrame with video_id including the +R or +L suffix where applicable
    """
    new_rows = []

    for index, row in df.iterrows():
        video_id = row['video_id'][:-2]
        print(video_id)
        # Skip rows where the video_id ends with a number suffix like '-1', '-2', etc.
        if re.search(r'-\d+$', video_id):
            continue

        # Check for +R and +L suffixes in the directory
        video_id_r = f"{video_id}-R.pkl"
        video_id_l = f"{video_id}-L.pkl"
        
        file_r = os.path.join(reference_directory, video_id_r)
        file_l = os.path.join(reference_directory, video_id_l)

        # If the +R file exists, add the row with the +R suffix
        if os.path.exists(file_r):
            updated_row = row.copy()
            updated_row['video_id'] = f"{video_id}-R"
            new_rows.append(updated_row)

        # If the +L file exists, add the row with the +L suffix
        if os.path.exists(file_l):
            updated_row = row.copy()
            updated_row['video_id'] = f"{video_id}-L"
            new_rows.append(updated_row)

    # Create a new DataFrame from the collected rows
    new_df = pd.DataFrame(new_rows)
    
    return new_df

def get_1h_2s_references(df_1, df_2s, reference_directory):
    """
    Searches for reference files (-R or -L suffix) for both df_1 and df_2s,
    and combines the results into a single DataFrame.

    Args:
        df_1 (pd.DataFrame): DataFrame for 1-handed data.
        df_2s (pd.DataFrame): DataFrame for 2-handed (same handshape) data.
        reference_directory (str): Path to the directory containing reference pose files.

    Returns:
        pd.DataFrame: Combined DataFrame with rows from df_1 and df_2s that have matching reference files.
    """
    # Get the reference rows for df_1
    df_1_references = search_reference_poses(df_1, reference_directory)
    
    # Get the reference rows for df_2s
    df_2s_references = search_reference_poses(df_2s, reference_directory)
    
    # Combine both DataFrames
    df_1_2s = pd.concat([df_1_references, df_2s_references], ignore_index=True)
    
    return df_1_2s

if __name__ == "__main__":
    # Load pose data
    # Example usage:

    json_file_path = 'PoseTools/data/metadata/output/all/ac_metadata.json'
    df = json_to_dataframe(json_file_path)
    
    # Assuming 'df' is the DataFrame you want to split
    df_1, df_2s, df_2a = split_dataframe_by_handedness(df)
    

    combined_df = get_1h_2s_references(df_1, df_2s, 'PoseTools/data/datasets/hamer_1_2s_2a/normalized')


    # Example usage
    poseconstructor = BuildReferencePose()
    reference_poses = poseconstructor.build_reference_pose(combined_df)
    print(reference_poses)
    import pickle
    # Save the object to a file
    with open('reference_poses.pkl', 'wb') as file:
        pickle.dump(reference_poses, file)
    exit()
    plot_multiple_hands_from_dict(reference_poses)
    exit()

    # Create an instance of the class
    evaluator = EvaluatePoses(df, reference_poses)

    # Call the evaluate_poses method to print video IDs
    evaluator.evaluate_poses()

