import os
import json
from tqdm import tqdm
from PoseTools.utils.processors import HamerProcessor, PklProcessor
from PoseTools.utils.parsers import PoseFormatParser, PklParser, HamerParser
from PoseTools.utils.preprocessing import PoseSelect
import matplotlib.pyplot as plt
import numpy as np


def plot_pose_frame(pose, edges, frame_idx=0, output_filename="pose_frame_plot.png"):
    """
    Plots a single frame of the pose data in x, y dimensions and saves the plot as a PNG file.

    Parameters:
    - pose: numpy array of shape [N_frames, nodes, dim=3]
    - edges: list of edges connecting the nodes
    - frame_idx: index of the frame to plot (default is 0)
    - output_filename: the filename to save the plot (default is "pose_frame_plot.png")
    """

    # Extract the x and y coordinates for the specified frame
    x = pose[frame_idx, :, 0]  # x coordinates
    y = pose[frame_idx, :, 1]  # y coordinates

    # Create a plot for the specified frame
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c='red', label='Keypoints')  # Scatter plot for the keypoints

    # Plot the edges between nodes
    for edge in edges:
        start, end = edge
        plt.plot([x[start], x[end]], [y[start], y[end]], 'b-', lw=2, label='_nolegend_')

    # Set axis limits and labels
    plt.xlim(min(x) - 0.1, max(x) + 0.1)
    plt.ylim(min(y) - 0.1, max(y) + 0.1)
    plt.title(f'Pose for Frame {frame_idx} (x, y Coordinates)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(output_filename)
    plt.close()

    print(f"Plot saved as {output_filename}")

# Example usage:
edges = [
    [2, 0], [1, 0], [0, 3], [0, 4], [3, 5], [4, 6], [5, 7], [6, 17],
    [7, 8], [7, 9], [9, 10], [7, 11], [11, 12], [7, 13], [13, 14], [7, 15],
    [15, 16], [17, 18], [17, 19], [19, 20], [17, 21], [21, 22], [17, 23], 
    [23, 24], [17, 25], [25, 26]
]

# Assuming pose is your actual pose array with shape [N_frames, nodes, dim=3]
pose_path = "PoseTools/data/datasets/mp/GLAS-B.pkl"
            
pose_loader = PklParser(pose_path)
pose, conf = pose_loader.read_pkl()
print(pose.shape)
plot_pose_frame(pose, edges, frame_idx=20, output_filename="PoseTools/data/converters/pose_frame_plot.png")


pose_path = "PoseTools/data/datasets/mp_pose/GLAS-B.pose"

pose_loader = PoseFormatParser(pose_path)
pose, conf = pose_loader.read_pose(n_points = 27)
pose = pose.squeeze(1)
plot_pose_frame(pose, edges, frame_idx=20, output_filename="PoseTools/data/converters/pose_frame_plot2.png")




'''
pose_path = "PoseTools/data/datasets/mp/GLAS-B.pkl"
            
pose_loader = PklParser(pose_path)
pose, conf = pose_loader.read_pkl()
print(pose.shape)

pose_path = "GMVISR/data/mp_old/05724.pkl"
            
pose_loader = PklParser(pose_path)
pose2, conf2 = pose_loader.read_pkl()
pose_select = [ 0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53, 54, 58, 59, 62, 63, 66, 67, 70, 71, 74]
pose2 = pose2[:, pose_select, :]


import matplotlib.pyplot as plt
import numpy as np

def plot_pose_frame(pose, edges, frame_idx=0, output_filename="pose_frame_plot.png"):
    """
    Plots a single frame of the pose data in x, y dimensions and saves the plot as a PNG file.

    Parameters:
    - pose: numpy array of shape [N_frames, nodes, dim=3]
    - edges: list of edges connecting the nodes
    - frame_idx: index of the frame to plot (default is 0)
    - output_filename: the filename to save the plot (default is "pose_frame_plot.png")
    """

    # Extract the x and y coordinates for the specified frame
    x = pose[frame_idx, :, 0]  # x coordinates
    y = pose[frame_idx, :, 1]  # y coordinates

    # Create a plot for the specified frame
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c='red', label='Keypoints')  # Scatter plot for the keypoints

    # Plot the edges between nodes
    for edge in edges:
        start, end = edge
        plt.plot([x[start], x[end]], [y[start], y[end]], 'b-', lw=2, label='_nolegend_')

    # Set axis limits and labels
    plt.xlim(min(x) - 0.1, max(x) + 0.1)
    plt.ylim(min(y) - 0.1, max(y) + 0.1)
    plt.title(f'Pose for Frame {frame_idx} (x, y Coordinates)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(output_filename)
    plt.close()

    print(f"Plot saved as {output_filename}")

# Example usage:
edges = [
    [2, 0], [1, 0], [0, 3], [0, 4], [3, 5], [4, 6], [5, 7], [6, 17],
    [7, 8], [7, 9], [9, 10], [7, 11], [11, 12], [7, 13], [13, 14], [7, 15],
    [15, 16], [17, 18], [17, 19], [19, 20], [17, 21], [21, 22], [17, 23], 
    [23, 24], [17, 25], [25, 26]
]

# Assuming pose is your actual pose array with shape [N_frames, nodes, dim=3]

plot_pose_frame(pose, edges, frame_idx=20, output_filename="pose_frame_plot.png")

plot_pose_frame(pose2, edges, frame_idx=0, output_filename="pose_frame_plot2.png")
'''