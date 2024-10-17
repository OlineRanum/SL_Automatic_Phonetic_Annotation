import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Define the edges you provided
edges = [
    [2, 0],
    [1, 0],
    [0, 3],
    [0, 4],
    [3, 5],
    [4, 6],
    [5, 7],
    [6, 17],
    [7, 8],
    [7, 9],
    [9, 10],
    [7, 11],
    [11, 12],
    [7, 13],
    [13, 14],
    [7, 15],
    [15, 16],
    [17, 18],
    [17, 19],
    [19, 20],
    [17, 21],
    [21, 22],
    [17, 23],
    [23, 24],
    [17, 25],
    [25, 26],
    [17, 7],
]

def plot_keypoints_with_edges(keypoints, edges, frame_idx=0):
    """
    Plots keypoints for a specific frame with the given edges.
    
    Args:
        keypoints (np.ndarray): Keypoints array of shape [T, N_keypoints, N_dims]
        edges (list): List of edges (pairs of indices)
        frame_idx (int): Frame index to plot (default is the first frame)
    """
    plt.figure(figsize=(8, 6))
    plt.title(f"Keypoints Visualization for Frame {frame_idx}")
    
    # Select the frame
    keypoints_frame = keypoints[frame_idx]
    
    # Extract x and y coordinates
    x_coords = keypoints_frame[:, 0]
    y_coords = keypoints_frame[:, 1]
    
    # Plot all keypoints
    plt.scatter(x_coords, y_coords, color='blue', s=50, zorder=2)
    
    # Plot edges
    for edge in edges:
        p1, p2 = edge
        plt.plot(
            [x_coords[p1], x_coords[p2]], 
            [y_coords[p1], y_coords[p2]], 
            'r-', zorder=1
        )
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().invert_yaxis()  # Invert Y-axis for better visualization
    plt.savefig("PoseTools/data/utils_data/keypoints_with_edges.png")

def load_pkl_file(file_path):
    """
    Loads a .pkl file and extracts keypoints.
    
    Args:
        file_path (str): Path to the .pkl file
    
    Returns:
        dict: Data loaded from the .pkl file
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":
    # Path to the .pkl file
    pkl_file_path = "PoseTools/data/datasets/wlasl_small/24726.pkl"

    # Load the pkl file
    data = load_pkl_file(pkl_file_path)

    # Extract keypoints (assuming keypoints are stored under 'keypoints')
    keypoints = np.array(data['keypoints'])

    # Plot keypoints with edges for the first frame (frame 0)
    plot_keypoints_with_edges(keypoints, edges, frame_idx=20)
