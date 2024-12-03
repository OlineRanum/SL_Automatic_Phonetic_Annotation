import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import numpy.ma as ma
from pose_format import Pose

def plot_frame_with_edges(pose, frame_idx=None, save_path=None):
    """
    Plot a single frame with keypoints and edges.

    :param pose: Numpy array of shape [num_frames, 1, num_keypoints, 3]
    :param edges: List of tuples indicating the edges (connections) between keypoints
    :param frame_idx: The index of the frame to plot
    :param save_path: Path to save the image, if None, the plot will be shown
    """
    edges =  [
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
                    ]
    
    fig, ax = plt.subplots()
    if frame_idx is None:
        frame = pose
    else:
        frame = pose[frame_idx, :, :]  # Get the frame data
    x = frame[:, 0]
    y = frame[:, 1]

    # Plot keypoints
    ax.scatter(x, y, color='blue')

    x_r = frame[16, 0]
    y_r = frame[16, 1]
    ax.scatter(x_r, y_r, color='red')

    x_l = frame[15, 0]
    y_l = frame[15, 1]
    ax.scatter(x_l, y_l, color='green')

    x_c = frame[0, 0]
    y_c = frame[0, 1]
    ax.scatter(x_c, y_c, color='k')


    # Plot edges
    #for edge in edges:
    #    start, end = edge
    #    ax.plot([x[start], x[end]], [y[start], y[end]], 'r-')

    # Add index numbers to the keypoints
    #for idx, (x_coord, y_coord) in enumerate(zip(x, y)):
    #    ax.text(x_coord, y_coord, str(idx), fontsize=12, color='black', ha='right')

    ax.set_aspect('equal', 'box')

    if save_path:
        plt.savefig(save_path)
        print(f"Frame saved to {save_path}")
    else:
        plt.show()
