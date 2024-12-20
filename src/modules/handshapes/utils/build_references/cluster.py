import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def cluster_frames(frames, gloss_labels, k):
    N, nodes, dims = frames.shape
    # Flatten each frame (nodes x dims) into a single vector (nodes * dims)
    data = frames.reshape(N, nodes * dims)
    # Optionally, scale the data
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # data = scaler.fit_transform(data)
    
    num_clusters = k

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(data)

    # Perform PCA for visualization
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    # Plot the clusters in PCA space
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.legend(*scatter.legend_elements(), title='Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('K-Means Clusters of Poses Projected onto PCA Components')
    plt.show()

    # Create a mapping from cluster labels to gloss labels
    clusters_to_glosses = {}
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        glosses_in_cluster = [gloss_labels[i] for i in indices]
        clusters_to_glosses[label] = glosses_in_cluster

    return labels, clusters_to_glosses

    


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_clustered_poses(frame_poses, labels, output_file_name, handshape='Handshape'):
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    
    # Initialize a dictionary to store average poses per cluster
    cluster_avg_poses = {}
    
    # Increase figure size for better visibility
    fig, axs = plt.subplots(num_clusters, 4, figsize=(20, 5 * num_clusters), subplot_kw={'projection': '3d'})
    angles = [[0, 0], [90, 90], [30, 30], [30, -30]]
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]

    fig.suptitle(f"Clustered Poses for {output_file_name}", fontsize=18)

    for idx, label in enumerate(unique_labels):
        # Collect all frames belonging to the current cluster
        cluster_frames = frame_poses[labels == label]
        avg_cluster_frame = np.mean(cluster_frames, axis=0)
        
        # Store the average pose in the dictionary
        cluster_avg_poses[label] = avg_cluster_frame
        
        # Use the same naming convention as in the JSON file
        cluster_title = f"{handshape}_{int(label) + 1}"
        
        for view_idx, angle in enumerate(angles):
            # Handle the case when num_clusters is 1
            if num_clusters == 1:
                ax = axs[view_idx]
            else:
                ax = axs[idx, view_idx]
            
            # Plot the average pose
            ax.scatter(avg_cluster_frame[:, 0], avg_cluster_frame[:, 1], avg_cluster_frame[:, 2], color='b', s=100)
            
            # Plot the connections between keypoints
            for connection in connections:
                start, end = connection
                xs = [avg_cluster_frame[start, 0], avg_cluster_frame[end, 0]]
                ys = [avg_cluster_frame[start, 1], avg_cluster_frame[end, 1]]
                zs = [avg_cluster_frame[start, 2], avg_cluster_frame[end, 2]]
                ax.plot(xs, ys, zs, color='r', linewidth=2)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlim(-0.1, 0.02)
            ax.set_xlim(-0.1, 0.15)
            ax.set_ylim(-0.1, 0.15)
            ax.set_zlabel('Z')
            ax.view_init(elev=angle[0], azim=angle[1])
            ax.set_title(f"Cluster {int(label)} - View {view_idx + 1}")
    
    plt.tight_layout()
    # Save the figure
    plt.savefig(f'{output_file_name}_{handshape}.png')
    #plt.show()
    
    # Return the dictionary of average poses
    return cluster_avg_poses


import json

def save_selected_clusters_to_json(cluster_avg_poses, selected_labels, output_file_name):
    """
    Save a subselection of cluster average poses to a JSON file.

    Parameters:
    - cluster_avg_poses: dict mapping cluster labels to average poses (NumPy arrays).
    - selected_labels: list of cluster labels to include in the output.
    - output_file_name: path to the JSON file to write.
    """
    # Initialize a dictionary to store the selected clusters
    selected_clusters = {}

    for label in selected_labels:
        if label in cluster_avg_poses:
            # Convert the NumPy array to a list for JSON serialization
            avg_pose_list = cluster_avg_poses[label].tolist()
            # Use string keys for JSON compatibility
            selected_clusters[str(label)] = avg_pose_list
        else:
            print(f"Warning: Cluster label {label} not found in cluster_avg_poses.")

    # Write the selected clusters to a JSON file
    with open(output_file_name, 'w') as json_file:
        json.dump(selected_clusters, json_file, indent = 1)

    print(f"Selected clusters saved to {output_file_name}")
    return selected_clusters

def get_unique_glosses_per_label(clusters_to_glosses, selected_labels):
    """
    For each cluster label in selected_labels, return the set of unique glosses present in that cluster.

    Parameters:
    - clusters_to_glosses: dict mapping cluster labels to lists of gloss labels.
    - selected_labels: list of cluster labels to process.

    Returns:
    - selected_clusters_glosses: dict mapping selected cluster labels to sets of unique glosses.
    """
    selected_clusters_glosses = {}
    
    for label in selected_labels:
        if label in clusters_to_glosses:
            selected_clusters_glosses[label] = clusters_to_glosses[label]
        else:
            print(f"Warning: Cluster label {label} not found in clusters_to_glosses.")
            selected_clusters_glosses[label] = set()
    
    for key, item in selected_clusters_glosses.items():
        print(f"Cluster {key} built on {len(item)} videos.")
        print(set(item))

    return selected_clusters_glosses
