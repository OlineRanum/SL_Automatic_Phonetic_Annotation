import json
import os
import pickle
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import Isomap

def compute_isomap_features(pose_dataset, n_components=10):
    """
    Apply Isomap to extract manifold-based features from hand poses.
    pose_dataset: A dataset of hand poses (shape [n_samples, n_keypoints, 3]).
    n_components: The number of dimensions to reduce to.
    """
    poses_flat = pose_dataset.reshape(pose_dataset.shape[0], -1)  # Flatten the poses
    isomap = Isomap(n_components=n_components)
    manifold_features = isomap.fit_transform(poses_flat)
    return manifold_features

# Placeholder for feature transformations
def calculate_distances(pose):
    """Calculate pairwise distances between keypoints."""
    n_keypoints = pose.shape[0]
    distances = torch.zeros(n_keypoints, n_keypoints)
    for i in range(n_keypoints):
        for j in range(i + 1, n_keypoints):
            distances[i, j] = torch.norm(pose[i] - pose[j])
            distances[j, i] = distances[i, j]
    return distances.triu().flatten()

def calculate_angles(pose):
    """Calculate angles between keypoints."""
    n_keypoints = pose.shape[0]
    angles = []
    for i in range(1, n_keypoints - 1):
        vec1 = pose[i] - pose[i - 1]
        vec2 = pose[i + 1] - pose[i]
        cos_theta = torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))
        angle = torch.acos(cos_theta)
        angles.append(angle.item())
    return torch.tensor(angles)

def normalize_to_wrist(pose):
    """Normalize keypoints relative to the wrist."""
    wrist = pose[0]
    normalized_pose = pose - wrist
    return normalized_pose

def compute_second_order_differences(features):
    """
    Compute second-order differences between first-order features.
    features: A tensor of first-order features (e.g., distances or angles).
    Returns: A flattened tensor of second-order differences.
    """
    differences = []
    n = features.size(0)
    for i in range(n):
        for j in range(i + 1, n):
            diff = features[i] - features[j]
            differences.append(diff.item())
    return torch.tensor(differences)

def build_graph_data(pose, transformation='raw', use_second_order=False):
    """Apply selected transformation to the keypoints and optionally add second-order differences."""
    if transformation == 'distances':
        first_order_features = calculate_distances(pose)
    elif transformation == 'angles':
        first_order_features = calculate_angles(pose)
    elif transformation == 'normalized':
        first_order_features = normalize_to_wrist(pose)
    else:
        first_order_features = pose.flatten()  # Default: raw positions

    if use_second_order:
        second_order_features = compute_second_order_differences(first_order_features)
        combined_features = torch.cat((first_order_features.flatten(), second_order_features.flatten()))
        return combined_features

    return first_order_features.flatten()

from mpl_toolkits.mplot3d import Axes3D

class FeatureSpaceExplorer:
    def __init__(self, data_dict, top_n):
        self.data_dict = data_dict
        self.top_n = top_n

    def explore_feature_space(self, transformation='raw', use_second_order=False):
        """Explore feature space by applying transformations and visualizing with t-SNE, PCA, or Isomap."""
        features, labels = [], []
        self.transformation = transformation
        
        # Extract features for each frame in the dataset
        for vid_id, data in tqdm(self.data_dict.items(), desc="Extracting features"):
            for frame_idx in range(data['node_pos'].shape[0]):
                pose = torch.tensor(data['node_pos'][frame_idx, :, :], dtype=torch.float32)
                closest_handshape = calculate_top_n_closest_handshapes(pose, n = self.top_n)
                if data['label'] not in closest_handshape:
                    continue
                    
                transformed_features = build_graph_data(pose, transformation=transformation, use_second_order=use_second_order)
                features.append(transformed_features.numpy())
                labels.append(data['label'])

        features = np.array(features)
        
        labels = np.array(labels)
        
        return features, labels

    def plot_tsne(self, features, labels, n_components=3):
        """Apply tplot_tsne-SNE and visualize the feature space in 3D with actual label names on the colorbar."""
        unique_labels = list(set(labels))  # Get unique labels
        label_to_numeric = {label: idx for idx, label in enumerate(unique_labels)}  # Map each label to an integer
        numeric_labels = [label_to_numeric[label] for label in labels]  # Convert labels to numeric

        print(features.shape)
        # Apply t-SNE to the features with 3 components
        tsne = TSNE(n_components=n_components, random_state=42)
        transformed_features = tsne.fit_transform(features)
        print(transformed_features.shape)
        exit()

        # Create a 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(transformed_features[:, 0], transformed_features[:, 1], transformed_features[:, 2], 
                             c=numeric_labels, cmap='viridis')

        # Attach a colorbar and set the tick labels to the actual label names
        cbar = fig.colorbar(scatter, ticks=range(len(unique_labels)))
        cbar.set_label('Handshape Class')
        cbar.set_ticks(range(len(unique_labels)))
        cbar.set_ticklabels(unique_labels)

        ax.set_title('3D t-SNE Projection of Handshape Features')
        plt.show()

        plt.savefig("/home/gomer/oline/PoseTools/src/modules/features/tsne_3d_"+str(self.transformation)+"_ncomp_"+str(self.top_n)+".png")

    def plot_pca(self, features, labels, n_components=3):
        """Apply PCA and visualize the feature space in 3D with actual label names on the colorbar."""
        unique_labels = list(set(labels))
        label_to_numeric = {label: idx for idx, label in enumerate(unique_labels)}
        numeric_labels = [label_to_numeric[label] for label in labels]

        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Apply PCA with 3 components
        pca = PCA(n_components=n_components)
        transformed_features = pca.fit_transform(features)

        # Create a 3D scatter plot
        scatter = ax.scatter(transformed_features[:, 0], transformed_features[:, 1], transformed_features[:, 2], 
                             c=numeric_labels, cmap='plasma')

        # Attach a colorbar and set the tick labels to the actual label names
        cbar = fig.colorbar(scatter, ticks=range(len(unique_labels)))
        cbar.set_label('Handshape Class')
        cbar.set_ticks(range(len(unique_labels)))
        cbar.set_ticklabels(unique_labels)

        ax.set_title('3D PCA Projection of Handshape Features')
        plt.show()

        plt.savefig("/home/gomer/oline/PoseTools/src/modules/features/pca_3d_"+str(self.transformation)+"_ncomp_"+str(self.top_n)+".png")

    def plot_isomap(self, features, labels, n_components=3):
        """Apply Isomap and visualize the feature space in 3D with actual label names on the colorbar."""
        
        plt.clf()
        fig = plt.figure()
        unique_labels = list(set(labels))  # Get unique labels
        label_to_numeric = {label: idx for idx, label in enumerate(unique_labels)}  # Map each label to an integer
        numeric_labels = [label_to_numeric[label] for label in labels]  # Convert labels to numeric

        # Apply Isomap with 3 components
        isomap = Isomap(n_components=n_components)
        transformed_features = isomap.fit_transform(features)

        # Create a 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(transformed_features[:, 0], transformed_features[:, 1], transformed_features[:, 2], 
                             c=numeric_labels, cmap='cividis')

        # Attach a colorbar and set the tick labels to the actual label names
        cbar = fig.colorbar(scatter, ticks=range(len(unique_labels)))
        cbar.set_label('Handshape Class')
        cbar.set_ticks(range(len(unique_labels)))
        cbar.set_ticklabels(unique_labels)

        ax.set_title(f'3D Isomap Projection ({n_components} Components)')
        plt.show()

        plt.savefig("/home/gomer/oline/PoseTools/src/modules/features/isomap_3d_"+str(self.transformation)+"_ncomp_"+str(self.top_n)+".png")



from PoseTools.src.modules.handedness.utils.graphics import read_dict_from_txt
gloss_mapping = read_dict_from_txt('/home/gomer/oline/PoseTools/data/metadata/output/global_value_to_id.txt')
with open('/home/gomer/oline/PoseTools/src/models/graphTransformer/utils/reference_poses.pkl', 'rb') as file:
        reference_poses = pickle.load(file)

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
    closest_handshape = gloss_mapping[int(keys[np.argmin(np.array(distances))])]
    return closest_handshape

def calculate_top_n_closest_handshapes(pose, n=3):
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
    for key, reference_pose in reference_poses.items():
        distance = np.linalg.norm(pose - reference_pose, axis=1).mean()
        distances.append(distance)
        keys.append(key)

    # Sort the distances and get the indices of the top `n` closest distances
    sorted_indices = np.argsort(distances)[:n]
    
    
    # Retrieve the corresponding handshapes
    top_n_handshapes = [gloss_mapping[int(keys[i])] for i in sorted_indices]
    
    return top_n_handshapes
    
def load_data(pickle_path, metadata_path):
    """Load the data from pickle and metadata, limiting to 10 examples per class."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    data_dict = {}
    class_counter = {}  # Dictionary to track the number of examples per class

    for item in metadata:
        handshape = item['instances'][0]['Handshape']  # Assuming Handshape is stored here
        
        # Initialize counter for this class (handshape)
        if handshape not in class_counter:
            class_counter[handshape] = 0

        for instance in item['instances']:
            if class_counter[handshape] >= 10:  # Stop when we have 10 examples for this class
                break

            vid_id = instance['video_id']
            source = instance['source']
            
            if os.path.exists(os.path.join(pickle_path, f'{vid_id}.pkl')):
                with open(os.path.join(pickle_path, f'{vid_id}.pkl'), 'rb') as pkl_file:
                    pose_data = pickle.load(pkl_file)
                    
                    if (source == 'Corpus') and (pose_data['keypoints'].shape[0] > 50):
                        continue

                    data_dict[vid_id] = {
                        'label': handshape,  # Use the handshape as the label
                        'node_pos': pose_data['keypoints'][:, :, :],  # Keypoints
                    }
                    class_counter[handshape] += 1  # Increment the counter for this class
                    if class_counter[handshape] >= 10:
                        break  # Stop adding more examples for this class after 10

    return data_dict

# Main function to run the exploration
if __name__ == "__main__":
    # Load data
    import os
    pickle_path = os.path.abspath("../../../../mnt/fishbowl/gomer/oline/hamer_pkl")
    metadata_path = "/home/gomer/oline/PoseTools/data/metadata/output/10c/10c_SB_train.json"
    
    data_dict = load_data(pickle_path, metadata_path)

    # Initialize feature space explorer
    explorer = FeatureSpaceExplorer(data_dict, top_n=3)

    # Choose a transformation (raw, distances, angles, normalized)
    transformations = ['raw', 'distances']#, 'angles']  # Example: Using pairwise distances as features
    for transformation in transformations:
        # Explore the feature space
        features, labels = explorer.explore_feature_space(transformation=transformation)

        # Visualize the feature space using t-SNE or PCA
        explorer.plot_tsne(features, labels)  # t-SNE visualization
        explorer.plot_pca(features, labels)  # PCA visualization
        explorer.plot_isomap(features, labels)  # PCA visualization
