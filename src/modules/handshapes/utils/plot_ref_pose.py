import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_hamer_hand_3d(node_positions, output_file_name):
    """
    Plots a 3D graph of hand nodes with edges between them.
    
    Parameters:
    - node_positions: A numpy array of shape (21, 3) representing the 3D positions of nodes.
    """
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index Finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle Finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring Finger
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky Finger
    ]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the nodes
    ax.scatter(node_positions[:, 0], node_positions[:, 1], node_positions[:, 2], color='b', s=100)

    # Plot the edges (connections)
    for start, end in connections:
        xs = [node_positions[start, 0], node_positions[end, 0]]
        ys = [node_positions[start, 1], node_positions[end, 1]]
        zs = [node_positions[start, 2], node_positions[end, 2]]
        ax.plot(xs, ys, zs, color='r', linewidth=2)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Save the plot
    plt.savefig(output_file_name)
    plt.close()

def process_directory(input_dir, output_dir):
    """
    Processes each .pkl file in the directory, creating a 3D plot for each.
    
    Parameters:
    - input_dir: Directory containing .pkl files.
    - output_dir: Directory to save the plot images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.pkl'):
            print(file_name)
            file_path = os.path.join(input_dir, file_name)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)['keypoints']
            print(data.shape)
            data = data[15]
            print(data.shape)
            # Assuming 'node_positions' is the key in the pickle file
            node_positions = data
            if node_positions is not None and node_positions.shape == (21, 3):
                output_file_name = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.png")
                plot_hamer_hand_3d(np.array(node_positions), output_file_name)
                print(f"Saved plot for {file_name} to {output_file_name}")

# Usage
input_directory = '/home/gomer/oline/PoseTools/src/modules/handshapes/utils/reference_hamer_pkl'
output_directory = '/home/gomer/oline/PoseTools/src/modules/handshapes/utils/img'
process_directory(input_directory, output_directory)
