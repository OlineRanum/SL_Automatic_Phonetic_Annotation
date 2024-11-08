import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_poses_from_json(json_file):
    # Load poses from JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Count number of poses (clusters) and print
    num_poses = len(data)
    print(f"Total number of poses (clusters): {num_poses}")
    
    # Define view angles for four different perspectives
    angles =[[0, 0], [90, 90], [30, 30], [30, -30]]
    
    # Create a figure with 4 subplots for each pose
    fig, axs = plt.subplots(num_poses, 4, figsize=(20, 5 * num_poses), subplot_kw={'projection': '3d'})
    
    for idx, (cluster_label, pose_data) in enumerate(data.items()):
        # Extract X, Y, Z coordinates from the pose data
        xs, ys, zs = zip(*pose_data)
        
        # Plot each pose from four different angles
        for view_idx, angle in enumerate(angles):
            ax = axs[idx, view_idx] if num_poses > 1 else axs[view_idx]
            
            # Plot the nodes
            ax.scatter(xs, ys, zs, color='b', s=100)
            
            # Connect points assuming hand structure
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index Finger
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle Finger
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring Finger
                (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky Finger
            ]
            
            for connection in connections:
                start, end = connection
                ax.plot([xs[start], xs[end]], [ys[start], ys[end]], [zs[start], zs[end]], color='r', linewidth=2)
            
            # Set view angle and labels
            ax.view_init(elev=angle[0], azim=angle[1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_zlim(-0.1, 0.02)
            ax.set_xlim(-0.1, 0.15)
            ax.set_ylim(-0.1, 0.15)
            ax.set_title(f"{cluster_label} - View {view_idx + 1}")

    plt.tight_layout()
    
    # Save the figure with an appropriate filename

    plt.savefig('/home/gomer/oline/PoseTools/src/modules/handshapes/utils/finals/finals_'+letter+'.png')
    plt.show()

letter = 'V'
# Example usage
json_file_path = '/home/gomer/oline/PoseTools/src/modules/handshapes/utils/finals/poses/'+letter+'_avg_pose.json'
plot_poses_from_json(json_file_path)
