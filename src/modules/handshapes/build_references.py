
import json, os, re
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm 
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
    def __init__(self, pose_dir, predefined_poses = None):
        self.pose_dir = pose_dir
        self.predefined_poses = predefined_poses
        self.handshape_dict = {}  # Dictionary to store handshape classes and their corresponding poses
            
        if self.predefined_poses is None:
            self.reference_poses = {}  # Dictionary to store the final reference poses
        else:
            self.reference_poses = predefined_poses
        self.gloss_mapping = read_dict_from_txt('/home/gomer/oline/PoseTools/data/metadata/output/global_value_to_id_large.txt')
    
    def preprocess(self, data, handshape):

        num_frames = data.shape[0]  # Total frames in the video
        
        #if (data['source'] == 'Corpus') & (num_frames > 25):
        #    continue  # Skip videos with more than 300 frames
        
        new_frames = []
        for frame_idx in range(num_frames):
            # Get the node positions (keypoint coordinates) for this frame
            
            pos = data[frame_idx, :, :]  # Shape [21, 3]
            #closest_handshape = self.calculate_euclidean_distance(pos.numpy())
            #if closest_handshape != strong_hand:
            #    continue
            closest_handshape = self.calculate_top_n_closest_handshapes(pos, n = 1)
            
            if handshape in closest_handshape:
                new_frames.append(pos)
        if not new_frames:
            return None
        else:
            return np.array(new_frames)

        

    def build_reference_pose(self, df):
        
        # Loop through each row and group poses by 'gloss' (handshape class)
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows", unit="row"):
  
            video_id = row['video_id']
            handshape = row['Handshape']
            filepath = os.path.join(self.pose_dir, f"{video_id}.pkl")
            try:
                with open(filepath, 'rb') as f:
                    data_dict = pickle.load(f)
            except FileNotFoundError:
                continue    
            # Extract keypoints data
            data = data_dict.get('keypoints', None)
            
            # Preprocess the data
            if self.predefined_poses is not None:
                data = self.preprocess(data, handshape)

            if data is None:
                continue
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


    def calculate_euclidean_distance(self, pose):
        """
        Calculates the Euclidean distance between a pose and a reference pose for each keypoint.
        
        Parameters:
        - pose: A numpy array of shape (21, 3), representing the pose for a frame.
        - reference_pose: A numpy array of shape (21, 3), representing the reference handshape pose.
        
        Returns:
        - The Euclidean distance between the pose and the reference pose.
        """
        distances = []
        keys = []
        for key, reference_pose in self.reference_poses.items():
            distances.append(np.linalg.norm(pose - reference_pose, axis=1).mean())
            keys.append(key)
        closest_handshape = self.gloss_mapping[int(keys[np.argmin(np.array(distances))])]
        return closest_handshape
    
    def calculate_top_n_closest_handshapes(self, pose, n=3):
        """
        Calculates the Euclidean distance between a pose and each reference pose, and returns the top `n` closest handshapes.
        
        Parameters:
        - pose: A numpy array of shape (21, 3), representing the pose for a frame.
        - n: The number of closest handshapes to return.
        
        Returns:
        - A list of the top `n` closest handshapes.
        """
        distances = []
        keys = []

        # Calculate the Euclidean distance between the pose and each reference pose
        for key, reference_pose in self.reference_poses.items():
            distance = np.linalg.norm(pose - reference_pose, axis=1).mean()
            distances.append(distance)
            keys.append(key)

        # Sort the distances and get the indices of the top `n` closest distances
        sorted_indices = np.argsort(distances)[:n]
        
        # Retrieve the corresponding handshapes
        top_n_handshapes = [self.gloss_mapping[int(keys[i])] for i in sorted_indices]
        
        return top_n_handshapes

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

def compare_dicts(dict_list, threshold=1e-6):
    # Initialize a list to store the differences between dictionaries
    dictionary_differences = []

    # Iterate through consecutive pairs of dictionaries
    for i in range(1, len(dict_list)):
        current_dict = dict_list[i]
        previous_dict = dict_list[i - 1]

        # Initialize a variable to accumulate differences for all keys in the dictionaries
        total_diff = 0

        # Ensure we are comparing dictionaries with the same keys
        for key in current_dict.keys():
            if key in previous_dict:
                diff_array = current_dict[key] - previous_dict[key]
                diff = np.linalg.norm(diff_array, axis=1).mean()
                
                # Only count differences above the threshold
                if diff > threshold:
                    total_diff += diff
        
        print('Total difference:', i, ', ', total_diff)
        # Store the total difference between the two dictionaries
        dictionary_differences.append(total_diff)
    

    # Plot the differences between consecutive dictionaries
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(1, len(dict_list)), dictionary_differences, marker='o')
    plt.title('Incremental Differences Between Consecutive Dictionaries')
    plt.xlabel('Dictionary Pair (1 vs 2, 2 vs 3, etc.)')
    plt.ylabel('Total Mean Euclidean Distance')
    plt.savefig('/home/gomer/oline/PoseTools/src/modules/handedness/graphics/comparison.png')


if __name__ == "__main__":
    # Load pose data
    # Example usage:

    json_file_path = 'PoseTools/data/metadata/output/35c/35c_SB_1_2s.json'
    pose_dir = '../../../../mnt/fishbowl/gomer/oline/hamer_pkl'
    
    df = json_to_dataframe(json_file_path)

    

    
    print('Performing first itteration')
    poseconstructor = BuildReferencePose(pose_dir)
    reference_poses_1 = poseconstructor.build_reference_pose(df)
    output_path = '/home/gomer/oline/PoseTools/src/modules/handedness/graphics/itteration_1.png'
    #plot_multiple_hands_from_dict(reference_poses_1, output_path=output_path)


    print('Performing second itteration')
    poseconstructor = BuildReferencePose(pose_dir, predefined_poses = reference_poses_1)
    reference_poses_2 = poseconstructor.build_reference_pose(df)
    output_path = '/home/gomer/oline/PoseTools/src/modules/handedness/graphics/itteration_2.png'
    #plot_multiple_hands_from_dict(reference_poses_2, output_path=output_path)

    print('Performing third itteration')
    poseconstructor = BuildReferencePose(pose_dir, predefined_poses = reference_poses_2)
    reference_poses_3 = poseconstructor.build_reference_pose(df)
    output_path = '/home/gomer/oline/PoseTools/src/modules/handedness/graphics/itteration_3.png'
    #plot_multiple_hands_from_dict(reference_poses_3, output_path=output_path)

    print('Performing fourth itteration')
    poseconstructor = BuildReferencePose(pose_dir, predefined_poses = reference_poses_3)
    reference_poses_4 = poseconstructor.build_reference_pose(df)
    output_path = '/home/gomer/oline/PoseTools/src/modules/handedness/graphics/itteration_4.png'
    plot_multiple_hands_from_dict(reference_poses_4, output_path=output_path)

    print('Performing fifth itteration')
    poseconstructor = BuildReferencePose(pose_dir, predefined_poses = reference_poses_4)
    reference_poses_5 = poseconstructor.build_reference_pose(df)
    output_path = '/home/gomer/oline/PoseTools/src/modules/handedness/graphics/itteration_5.png'
    plot_multiple_hands_from_dict(reference_poses_5, output_path=output_path)


    # Save dictionary to a text file
    import pickle

    # Save the dictionary as a pickle file
    with open('/home/gomer/oline/PoseTools/src/modules/handedness/utils/references/reference_poses_it5.pkl', 'wb') as file:
        pickle.dump(reference_poses_5, file)

    dict_list = [reference_poses_1, reference_poses_2, reference_poses_3, reference_poses_4, reference_poses_5]
    compare_dicts(dict_list)
