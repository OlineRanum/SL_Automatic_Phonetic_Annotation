import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_clustered_poses(frame_poses, labels, output_file_name, handshape='Handshape', output_dir = ''):
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
        
        # Store the average pose in the dictionary
        cluster_avg_poses[label] = frame_poses[idx]
        
        # Use the same naming convention as in the JSON file
        cluster_title = f"{handshape}_{int(label) + 1}"
        avg_cluster_frame = cluster_frames 

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
    plt.savefig(f'{output_dir}/{output_file_name}_{handshape}.png')
    #plt.show()
    
    # Return the dictionary of average poses
    return cluster_avg_poses


def process_and_plot_json_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            
            # Load JSON data
            with open(filepath, 'r') as file:
                data = json.load(file)
            
            # Extract frames and poses
            frame_poses = []
            labels = []
            
            for label, poses in data.items():

                poses = np.array(poses)
                label = int(label)  # Convert string keys to integer labels
                labels.extend([label] * len(poses))  # Duplicate the label for each frame
                frame_poses.extend(poses)  # Collect all poses

            # Convert to numpy arrays for further processing
            frame_poses = np.array(frame_poses)
            labels = np.array(labels)
            
            # Define output file name based on the input filename
            output_file_name = os.path.splitext(filename)[0]
            
            # Call the plotting function
            plot_clustered_poses(frame_poses, labels, output_file_name, handshape='', output_dir='references/poses/figs')

# Replace 'your_directory_path' with the path to the directory containing the JSON files
process_and_plot_json_files('references/poses')
