from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
import pandas as pd
from PoseTools.src.modules.handedness.utils.graphics import read_dict_from_txt, plot_hamer_hand_3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

class BuildReferencePose:
    def __init__(self, pose_dir, n_clusters = 2, handshape = 'S', predefined_poses=None, gloss_map_path='/home/gomer/oline/PoseTools/data/metadata/output/global_value_to_id.txt', itt=1):
        self.itt = itt
        self.handshape = handshape
        self.pose_dir = pose_dir
        self.predefined_poses = predefined_poses
        self.pose_dict = {}
        self.n_clusters = n_clusters
        self.gloss_mapping = read_dict_from_txt(gloss_map_path)

    def pairwise_distance_matrix(self, points):
        num_points = points.shape[0]
        dist_matrix = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(i, num_points):
                dist = np.linalg.norm(points[i] - points[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        max_distance = np.max(dist_matrix)
        return dist_matrix / max_distance if max_distance > 0 else dist_matrix

    def build_reference_pose(self, df):
        frame_matrices = []
        frame_poses = []
        gloss_labels = []

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            video_id = row['video_id']
            filepath = os.path.join(self.pose_dir, f"{video_id}.pkl")
            try:
                with open(filepath, 'rb') as f:
                    data_dict = pickle.load(f)
            except FileNotFoundError:
                continue
            data = data_dict.get('keypoints')
            n_frames = data.shape[0]

            # Calculate the start and end indices for the middle third
            start_index = n_frames // 3
            end_index = 2 * n_frames // 3
            
            # Slice the data to get the middle third
            data = data[start_index:end_index, :, :]
            if data is None:
                continue

            for frame in data:
                frame_pdm = self.pairwise_distance_matrix(frame)
                frame_matrices.append(frame_pdm.flatten())
                frame_poses.append(frame)
                gloss_labels.append(row['gloss'])

        pose_vectors = np.array(frame_matrices)
        labels = self.cluster_frames(pose_vectors)

        self.num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        num_outliers = list(labels).count(-1)
        total_datapoints = len(pose_vectors)

        print(f"Total number of datapoints: {total_datapoints}")
        print(f"Number of clusters: {self.num_clusters}")
        print(f"Number of outliers: {num_outliers}")
                
        avg_poses, avg_pdms = self.calculate_average_clusters(frame_poses, labels)
        self.save_cluster_data("/home/gomer/oline/PoseTools/src/modules/handshapes/utils/clusters/"+self.handshape+"_avg_pose.json", "/home/gomer/oline/PoseTools/src/modules/handshapes/utils/clusters/"+self.handshape+"_avg_pdm.json", avg_poses, avg_pdms)
        
        self.plot_clustered_poses(frame_poses, labels, gloss_labels, "clustered_poses")
        self.plot_cluster_scatter(pose_vectors, labels, 'clustered', method='PCA')

    def calculate_average_clusters(self, frame_poses, labels):
        unique_labels = np.unique(labels)
        avg_poses = {}
        avg_pdms = {}
        
        for idx, label in enumerate(unique_labels):
            cluster_frames = [frame_poses[i] for i in range(len(labels)) if labels[i] == label]
            avg_pose = np.mean(cluster_frames, axis=0)
            avg_pdm = self.pairwise_distance_matrix(avg_pose)
            
            # Use handshape as prefix for each cluster label
            cluster_label = f"{self.handshape}_{idx + 1}"
            avg_poses[cluster_label] = avg_pose.tolist()
            avg_pdms[cluster_label] = avg_pdm.tolist()
        
        return avg_poses, avg_pdms

    def save_cluster_data(self, pose_file, pdm_file, avg_poses, avg_pdms):
        """
        Saves the average poses and pairwise distance matrices for each cluster to JSON files.
        """
        with open(pose_file, 'w') as pf:
            json.dump(avg_poses, pf, indent=1)  # Add indent for new lines
        with open(pdm_file, 'w') as pf:
            json.dump(avg_pdms, pf, indent=1)  # Add indent for new lines

    
    def cluster_frames(self, pose_vectors):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        labels = kmeans.fit_predict(pose_vectors)
        return labels
    '''
    def cluster_frames(self, pose_vectors):
        # Use DBSCAN for density-based clustering
        print(pose_vectors.shape)
        eps = 0.4
        min_samples = 30
        print('pv shape ', pose_vectors.shape)
        print('Eps:', eps, '  Min Samples:', min_samples)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # Adjust eps and min_samples as needed
        labels = dbscan.fit_predict(pose_vectors)
        return labels
    '''    
    def plot_clustered_poses(self, frame_poses, labels, gloss_labels, output_file_name):
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels)
        
        # Increase figure size for better visibility
        fig, axs = plt.subplots(num_clusters, 4, figsize=(20, int(5*self.num_clusters)), subplot_kw={'projection': '3d'})
        angles = [[0, 0], [30, 30], [60, 60], [90, 90]]
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]

        fig.suptitle(f"Clustered Poses for {output_file_name}", fontsize=18)

        for idx, label in enumerate(unique_labels):
            cluster_frames = [frame_poses[i] for i in range(len(labels)) if labels[i] == label]
            avg_cluster_frame = np.mean(cluster_frames, axis=0)
            
            # Use the same naming convention as in the JSON file
            cluster_title = f"{self.handshape}_{idx + 1}"
            
            for view_idx, angle in enumerate(angles):
                ax = axs[idx, view_idx]
                ax.scatter(avg_cluster_frame[:, 0], avg_cluster_frame[:, 1], avg_cluster_frame[:, 2], color='b', s=100)
                
                for connection in connections:
                    start, end = connection
                    xs = [avg_cluster_frame[start, 0], avg_cluster_frame[end, 0]]
                    ys = [avg_cluster_frame[start, 1], avg_cluster_frame[end, 1]]
                    zs = [avg_cluster_frame[start, 2], avg_cluster_frame[end, 2]]
                    ax.plot(xs, ys, zs, color='r', linewidth=2)

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.view_init(elev=angle[0], azim=angle[1])
                ax.set_title(cluster_title)  # Set title as cluster label

        plt.tight_layout()

        plt.savefig(f'/home/gomer/oline/PoseTools/src/modules/handshapes/utils/clusters/{output_file_name}_{self.handshape}.png')
        plt.show()        
    
    def save_to_json(self, filename, data):
        """
        Save data to a JSON file.
        
        Parameters:
        - filename: The name of the JSON file.
        - data: The data to be saved, typically in dictionary form.
        """
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        
        
    def plot_cluster_scatter(self, pose_vectors, labels, output_file_name, method='PCA'):
        """
        Plots a 2D scatter plot of the clustered poses after dimensionality reduction.
        
        Parameters:
        - pose_vectors: Array of shape (n_instances, 441), flattened pairwise distance matrices.
        - labels: Array of cluster labels for each pose.
        - output_file_name: The name to save the scatter plot image file.
        - method: The dimensionality reduction method ('PCA' or 'TSNE').
        """
        # Dimensionality reduction
        if method == 'PCA':
            reducer = PCA(n_components=2)
        elif method == 'TSNE':
            reducer = TSNE(n_components=2, random_state=0)
        else:
            raise ValueError("Method should be 'PCA' or 'TSNE'")
        
        reduced_data = reducer.fit_transform(pose_vectors)

        # Plotting the scatter plot with different alpha for outliers
        plt.figure(figsize=(8, 6))



        # Plot outliers with reduced alpha
        outlier_mask = labels == -1
        scatter_outliers = plt.scatter(
            reduced_data[outlier_mask, 0], 
            reduced_data[outlier_mask, 1], 
            c="gray",  # Use a neutral color for outliers
            s=15, 
            alpha=0.25
        )
        # Plot in-cluster points with normal alpha
        in_cluster_mask = labels != -1
        scatter_in_cluster = plt.scatter(
            reduced_data[in_cluster_mask, 0], 
            reduced_data[in_cluster_mask, 1], 
            c=labels[in_cluster_mask], 
            cmap='viridis', 
            s=15, 
            alpha=1.0
        )


        # Add colorbar and labels
        plt.colorbar(scatter_in_cluster, label="Cluster Label")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.title(f"{method} Scatter Plot of Clusters")

        # Save and show the plot
        plt.savefig(f'/home/gomer/oline/PoseTools/src/modules/handshapes/utils/clusters/{output_file_name}_{self.handshape}_scatter.png')
        plt.show()

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def json_to_dataframe(json_file, handshape):
    with open(json_file, 'r') as f:
        gloss_data = json.load(f)
    data = []
    for gloss_entry in gloss_data:
        gloss = gloss_entry['gloss']
        instances = gloss_entry['instances']
        for instance in instances:
            instance['gloss'] = gloss
            data.append(instance)
    df = pd.DataFrame(data)
    df = df[df['Sign Type'] == '2s']
    df = df[df['Handshape'] == handshape]
    return df


if __name__ == "__main__":
    import json
    json_file_path = '/home/gomer/oline/PoseTools/data/metadata/output/50c/50c_uva.json'
    #json_file_path = '/home/gomer/oline/PoseTools/data/metadata/output/35c/35c_test.json'
    
    pose_dir = '../../../../mnt/fishbowl/gomer/oline/sb_uva/hamer_pkl'
    #json_file_path = '/home/gomer/oline/PoseTools/data/metadata/output/35c/35c_SB_1_2s_2a.json'
    #pose_dir = '../../../../mnt/fishbowl/gomer/oline/hamer_pkl'
    
    handshape = 'L'
    k = 8
    df = json_to_dataframe(json_file_path, handshape = handshape)

    poseconstructor = BuildReferencePose(pose_dir, n_clusters=k, handshape = handshape)
    poseconstructor.build_reference_pose(df)
    
