
import json, os
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm 
from PoseTools.src.modules.handedness.utils.graphics import read_dict_from_txt, plot_multiple_hands_from_dict, plot_hamer_hand_3d
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt

class BuildReferencePose:
    def __init__(self, pose_dir, predefined_poses = None, gloss_map_path = '/home/gomer/oline/PoseTools/data/metadata/output/global_value_to_id.txt', itt = 1):
        self.itt = itt
        self.pose_dir = pose_dir
        self.predefined_poses = predefined_poses
        self.handshape_dict = {}  # Dictionary to store handshape classes and their corresponding poses
        self.pose_dict = {}
        if self.predefined_poses is None:
            self.reference_pdms = {}
            self.reference_poses = {}  # Dictionary to store the final reference poses
        else:
            self.reference_pdms = predefined_poses
            self.reference_poses = predefined_poses
        self.gloss_mapping = read_dict_from_txt(gloss_map_path)
    
    def preprocess(self, data, handshape):

        num_frames = data.shape[0]  # Total frames in the video
        
        #if (data['source'] == 'Corpus') & (num_frames > 25):
        #    continue  # Skip videos with more than 300 frames
        
        new_frames = []
        for frame_idx in range(num_frames):
            # Get the node positions (keypoint coordinates) for this frame
            
            pos = data[frame_idx, :, :]  # Shape [21, 3]
            #closest_handshape = self.get_handshape(pos.numpy())
            #if closest_handshape != strong_hand:
            #    continue
            closest_handshape = self.calculate_top_n_closest_handshapes(pos, n = 1)
            
            if handshape in closest_handshape:
                new_frames.append(pos)
        if not new_frames:
            return None
        else:
            return np.array(new_frames)

    def pairwise_distance_matrix(self, points):
        """
        Calculate the normalized symmetric pairwise distance matrix for a given 3D point cloud.
        
        Parameters:
            points (numpy.ndarray): An array of shape (21, 3) representing the 3D coordinates of the hand nodes.
            
        Returns:
            numpy.ndarray: A normalized symmetric pairwise distance matrix of shape (21, 21).
        """

        # Number of nodes
        num_points = points.shape[0]
        
        # Initialize the distance matrix
        dist_matrix = np.zeros((num_points, num_points))
        
        # Calculate pairwise Euclidean distances manually
        for i in range(num_points):
            for j in range(i, num_points):  # Calculate only upper triangle
                dist = np.linalg.norm(points[i] - points[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist  # Ensure symmetry
        
        # Normalize by the maximum distance
        max_distance = np.max(dist_matrix)
        if max_distance > 0:
            normalized_dist_matrix = dist_matrix / max_distance
        else:
            normalized_dist_matrix = dist_matrix  # If max distance is 0, avoid division

        return normalized_dist_matrix

    
    

    def plot_pairwise_distance_matrix(self, matrix, gloss, title="Normalized Pairwise Distance Matrix "):
        """
        Plots the normalized pairwise distance matrix as a heatmap.
        
        Parameters:
            matrix (numpy.ndarray): The normalized pairwise distance matrix of shape (21, 21).
            title (str): The title for the plot.
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Normalized Distance')
        plt.title(title + gloss)
        plt.xlabel("Node Index")
        plt.ylabel("Node Index")
        plt.savefig('/home/gomer/oline/PoseTools/src/modules/handshapes/utils/pdm/' + gloss + '_' +self.itt +'.png')


    def calculate_quaternions(self, keypoints):
        """
        Given a (21, 3) matrix of keypoints in 3D, calculate the quaternion representing
        the rotation for each segment between consecutive keypoints.
        
        Parameters:
        - keypoints: np.ndarray of shape (21, 3), representing 3D keypoints.
        
        Returns:
        - quaternions: List of quaternions (as [w, x, y, z] arrays) for each segment.
        """
        # List to store quaternions for each segment
        quaternions = []
        
        # Define a reference vector (e.g., pointing along x-axis)
        reference_vector = np.array([1, 0, 0])
        
        # Loop through each consecutive pair of keypoints
        for i in range(len(keypoints) - 1):
            # Calculate the direction vector for the current segment
            segment_vector = keypoints[i + 1] - keypoints[i]
            segment_vector /= np.linalg.norm(segment_vector)  # Normalize it

            # Calculate rotation quaternion from reference vector to segment vector
            rotation = R.from_rotvec(np.cross(reference_vector, segment_vector))
            quaternion = rotation.as_quat()  # Format: [x, y, z, w]
            
            # Reorder to [w, x, y, z] and append to list
            quaternions.append([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])

        return np.array(quaternions)

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
                print(f"File not found: {filepath}")
                continue    
            
            # Extract keypoints data
            data = data_dict.get('keypoints', None)
            
            # Preprocess the data if predefined poses are provided
            if self.predefined_poses is not None:
                data = self.preprocess(data, handshape)
            print(data)
            if data is None:
                continue
            
            # Average across time steps (axis=0), resulting in [n_nodes, spatial_dims]
            avg_pose = data.mean(axis=0)
            
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('Average pose', avg_pose.shape)
            avg_pose_pdm = self.pairwise_distance_matrix(avg_pose)
            
            # Get the handshape class (gloss)
            gloss = row['gloss']
            #gloss = self.gloss_mapping[int(gloss)]
            
            # Initialize lists in dictionaries if gloss does not exist
            if gloss not in self.handshape_dict:
                self.handshape_dict[gloss] = []
            if gloss not in self.pose_dict:
                self.pose_dict[gloss] = []

            # Append the average pose and PDM to the lists
            self.handshape_dict[gloss].append(avg_pose_pdm)
            self.pose_dict[gloss].append(avg_pose)
        
        # Initialize reference dictionaries with lists containing the average poses
        total_poses = 0
        for gloss, poses in self.handshape_dict.items():
            poses = np.array(poses)
            avg_class_pose_pdm = poses.mean(axis=0)
            self.reference_pdms[gloss] = [np.array(avg_class_pose_pdm)]  # Initialize as a list with the avg pose as the first element
            total_poses += len(poses)

        for gloss, poses in self.pose_dict.items():
            
            poses = np.array(poses)
            avg_class_pose = poses.mean(axis=0)
            self.reference_poses[gloss] = [np.array(avg_class_pose)]  # Initialize as a list with the avg pose as the first element
            plot_hamer_hand_3d(avg_class_pose, gloss)
            self.plot_pairwise_distance_matrix(avg_class_pose, gloss=gloss)

        print(f"Total number of poses: {total_poses}")
        return self.reference_poses, self.reference_pdms
    

            

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


    def get_handshape(self, pose):
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
        pose = self.pairwise_distance_matrix(pose)
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
        #pose = self.calculate_quaternions(pose)
        pose = self.pairwise_distance_matrix(pose)
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

def json_to_dataframe(json_file):
    # Load the JSON data from the file
    with open(json_file, 'r') as f:
        gloss_data = json.load(f)

    # Create a list to hold the instance data
    data = []

    # Iterate over gloss entries
    for gloss_entry in gloss_data:
        gloss = gloss_entry['gloss'] 
        instances = gloss_entry['instances']  

        # Iterate over each instance and append the gloss to the instance details
        for instance in instances:
            instance['gloss'] = gloss  
            data.append(instance)  

    # Convert the list of instance dictionaries into a pandas DataFrame
    df = pd.DataFrame(data)
    print(df.keys())
    df = df[df['Sign Type'] == '2s']

    return df

# Function to add new data from JSON files into reference_pdms
def add_json_to_reference_pdms(reference_pdms, dir_pdms):
    for filename in os.listdir(dir_pdms):
        if filename.endswith('.json'):
            base_key = filename.split('_avg')[0]  # Get the base key from the filename
            file_path = os.path.join(dir_pdms, filename)
            
            with open(file_path, 'r') as f:
                new_data = json.load(f)

                # Loop through each key-value pair in the JSON data
                for version_key, value in new_data.items():
                    value = np.array(value)  # Convert to numpy array
                    # Check if base_key exists in reference_pdms
                    if base_key in reference_pdms:
                        # Append the new version data to the existing list for this base key
                        if isinstance(reference_pdms[base_key], list):
                            reference_pdms[base_key].append(value)
                        else:
                            # Convert to list if it's a single item
                            reference_pdms[base_key] = [reference_pdms[base_key], value]
                    else:
                        # Initialize as a list with the new version data
                        reference_pdms[base_key] = [value]

if __name__ == "__main__":
    # Load pose data 
    json_file_path = '/home/gomer/oline/PoseTools/data/metadata/output/50c/50c_uva.json'
    pose_dir = '../../../../mnt/fishbowl/gomer/oline/sb_uva/hamer'
    df = json_to_dataframe(json_file_path)
    df = df[df['Sign Type'] == '2s']
    print(len(df))
    print(len(df['Handshape'].unique()))
    
    print(df['Handshape'].unique()) 
    
    add_detailed = True

    
    # Define the number of iterations
    num_iterations = 1

    # Loop through the iterations
    for i in range(1, num_iterations + 1):
        print(f'Performing iteration {i}')
        
        # Build the reference pose for the current iteration
        if i == 1:
            poseconstructor = BuildReferencePose(pose_dir, itt = str(i))
        else:
            poseconstructor = BuildReferencePose(pose_dir, predefined_poses=predefined_poses, itt=str(i))
            
        reference_poses, reference_pdms = poseconstructor.build_reference_pose(df)
        
        # Update the predefined poses for the next iteration
        predefined_poses = reference_pdms

        if i == num_iterations:
            output_path = f'/home/gomer/oline/PoseTools/src/modules/handshapes/utils//build_references/references/iteration_{i}_pdm_uva.png'
            
            save_path_pdm = f'/home/gomer/oline/PoseTools/src/modules/handshapes/utils/build_references/references/reference_poses_pdm.pkl'
            save_path_raw = f'/home/gomer/oline/PoseTools/src/modules/handshapes/utils/build_references/references/reference_poses_raw.pkl'

            if add_detailed:
                dir_pdms = '/home/gomer/oline/PoseTools/src/modules/handshapes/utils/build_references/reference_data/finals/pdm'
                keys_to_drop = ['5m','5m_closed','W', 'V', 'Baby_O', 'T_open' , 'O',
                                 'Baby_beak', 'Baby_beak_open', 'Beak_open_spread', 'Beak_open', 'Beak_spread', 
                                 'N','M', '5r', 'Y', 'I', 'L', '1', '1_curved', '3', '4', '5', '5r' , 'A', 'B', 'C', 'C_spread', 'S', 'Baby_C' 'Y', 'V_curved']

                # Drop the specified keys from reference_pdms
                for key in keys_to_drop:
                    reference_pdms.pop(key, None)  # pop removes the key if it exists, and does nothing if it doesn't
                add_json_to_reference_pdms(reference_pdms, dir_pdms)
                '''print('\nPDMs:')
                for key, poses in reference_pdms.items():
                    print(f"Key: {key}")
                    for i, pose in enumerate(poses):
                        # Assuming each pose is a numpy array, you can use `.shape`
                        # Otherwise, replace `.shape` with a suitable method to get the data structure's dimensions
                        print(f"  PDM {i + 1} shape: {pose.shape if hasattr(pose, 'shape') else 'Shape not available'}")
                '''
                print('\nPoses:')
                for key, poses in reference_poses.items():
                    print(f"Key: {key}")
                    for i, pose in enumerate(poses):
                        # Assuming each pose is a numpy array, you can use `.shape`
                        # Otherwise, replace `.shape` with a suitable method to get the data structure's dimensions
                        print(f"  Pose {i + 1} shape: {pose.shape if hasattr(pose, 'shape') else 'Shape not available'}")

            with open(save_path_pdm, 'wb') as file:
                pickle.dump(reference_pdms, file)
            with open(save_path_raw, 'wb') as file:
                pickle.dump(reference_poses, file)
            
            
                
            #plot_multiple_hands_from_dict(reference_poses, output_path=output_path)
