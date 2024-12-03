
import matplotlib.pyplot as plt 

def plot_hamer_hand_3d(node_positions, output_file_name, normal_vector, wrist_idx = 0, index_base = 5, pinky_base = 17):
    """
    Plots a 3D graph of hand nodes with edges between them.
    
    Parameters:
    - node_positions: A numpy array of shape (21, 3) representing the 3D positions of nodes.
    """
    # Connections representing edges between the nodes
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index Finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle Finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring Finger
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky Finger
    ]
    wrist = node_positions[wrist_idx]
    index = node_positions[index_base]
    pinky = node_positions[pinky_base]
    
    # Create a 2x2 grid for subplots
    fig = plt.figure(figsize=(12, 12))
    angles = [[0, 0], [30, -30], [30, -60], [90, 90]]

    fig.suptitle(f"3D Handshape Visualization for Gloss: {output_file_name}", fontsize=16)
    print(node_positions[pinky_base, 0], node_positions[index_base, 0])
    print(node_positions[pinky_base, 1], node_positions[index_base, 1])
    print(node_positions[pinky_base, 2], node_positions[index_base, 2])
    for i, angle in enumerate(angles):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        
        # Plot the nodes
        ax.scatter(node_positions[:, 0], node_positions[:, 1], node_positions[:, 2], color='b', s=100)
        ax.scatter(node_positions[0, 0], node_positions[0, 1], node_positions[0, 2], color='r', s=100)
        ax.scatter(node_positions[index_base, 0], node_positions[index_base, 1], node_positions[index_base, 2], color='r', s=100)
        ax.scatter(node_positions[pinky_base, 0], node_positions[pinky_base, 1], node_positions[pinky_base, 2], color='r', s=100)
        
        # Plot the edges (connections)
        for connection in connections:
            start, end = connection
            xs = [node_positions[start, 0], node_positions[end, 0]]
            ys = [node_positions[start, 1], node_positions[end, 1]]
            zs = [node_positions[start, 2], node_positions[end, 2]]
            ax.plot(xs, ys, zs, color='r', linewidth=2)
        
        ax.quiver(
            wrist[0], wrist[1], wrist[2],
            normal_vector[0], normal_vector[1], normal_vector[2],
            color='purple', label='Palm Normal Vector', linewidth=3, length=0.5, arrow_length_ratio=0.2
        )

        # Set labels and viewing angle
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=angle[0], azim=angle[1])
        
        ax.set_title(f"View angle: {angle}°")

    # Save the plot as a single image
    plt.tight_layout()
    #plt.savefig(f'PoseTools/src/modules/handshapes/utils/reference_poses/{output_file_name}.png')
    plt.show()




def plot_hand_3d(ax, node_positions, gloss):
    """
    Plots a 3D graph of hand nodes with edges between them on a given axis.
    
    Parameters:
    - ax: The subplot axis to plot on.
    - node_positions: A numpy array of shape (21, 3) representing the 3D positions of nodes.
    - gloss: Title for the subplot, which will be the gloss (label).
    """
    # Connections representing edges between the nodes
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index Finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle Finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring Finger
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky Finger
    ]
    
    node_positions = node_positions[0]
    # Plot the nodes
    ax.scatter(node_positions[:, 0], node_positions[:, 1], node_positions[:, 2], color='b', s=20)

    # Plot the edges (connections)
    for connection in connections:
        start, end = connection
        xs = [node_positions[start, 0], node_positions[end, 0]]
        ys = [node_positions[start, 1], node_positions[end, 1]]
        zs = [node_positions[start, 2], node_positions[end, 2]]
        ax.plot(xs, ys, zs, color='r', linewidth=1)

    # Set labels
    ax.set_title(gloss, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_gif(pose_left, pose_right, gif_path='/home/gomer/oline/PoseTools/src/modules/orientation/handshape.gif'):
    """
    Creates a GIF of the 3D handshapes for both hands over time from three different viewing angles.
    
    Parameters:
    - pose_left: numpy array of shape [T, n_nodes, 3] for the left hand.
    - pose_right: numpy array of shape [T, n_nodes, 3] for the right hand.
    - gif_path: Path to save the output GIF.
    """
    # Angles for viewing the 3D plots
    angles = [[0, 0], [30, -30], [90, 90]]

    # Define edges connecting keypoints for plotting
    inward_edges = [
        [1, 0], [2, 1], [3, 2], [4, 3],     # Thumb
        [5, 0], [6, 5], [7, 6], [8, 7],     # Index Finger
        [9, 0], [10, 9], [11, 10], [12, 11],# Middle Finger
        [13, 0], [14, 13], [15, 14], [16, 15],# Ring Finger
        [17, 0], [18, 17], [19, 18], [20, 19] # Pinky Finger
    ]

    T, n_nodes, _ = pose_right.shape

    # Initialize the figure and axes
    fig = plt.figure(figsize=(12, 8))
    axes = []
    scatters = []
    line_collections = []

    # Create subplots for both hands
    for row, pose in enumerate([pose_right, pose_left]):
        for idx_angle, angle in enumerate(angles):
            ax = fig.add_subplot(2, 3, row * 3 + idx_angle + 1, projection='3d')
            ax.set_title(f'{"Right" if row == 0 else "Left"} Hand - View {idx_angle + 1}')
            ax.set_xlim(-0.2, 0.2)
            ax.set_ylim(-0.2, 0.2)
            ax.set_zlim(-0.1, 0.2)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=angle[0], azim=angle[1])

            # Add scatter and lines for this view
            scatter = ax.scatter([], [], [], c='b', s=20)
            lines = [ax.plot([], [], [], 'r-', linewidth=1)[0] for _ in inward_edges]

            axes.append(ax)
            scatters.append(scatter)
            line_collections.append(lines)

    # Update function for the animation
    def update(frame):
        for row, pose in enumerate([pose_right, pose_left]):
            current_keypoint = pose[frame]
            x, y, z = current_keypoint[:, 0], current_keypoint[:, 1], current_keypoint[:, 2]

            # Update scatter points and lines for each view
            for i in range(3):  # 3 angles per hand
                idx = row * 3 + i
                scatters[idx]._offsets3d = (x, y, z)
                for line, (start, end) in zip(line_collections[idx], inward_edges):
                    line.set_data([x[start], x[end]], [y[start], y[end]])
                    line.set_3d_properties([z[start], z[end]])

        return scatters + [line for lines in line_collections for line in lines]

    # Create animation
    anim = FuncAnimation(fig, update, frames=T, interval=100, blit=False)

    # Save animation as GIF
    writer = PillowWriter(fps=3)
    anim.save(gif_path, writer=writer)

    plt.close(fig)
    print(f"GIF saved at {gif_path}")


import os
import imageio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_gif_with_normals(pose_left, pose_right, normal_vectors_left, normal_vectors_right,
                          gif_path='/home/gomer/oline/PoseTools/src/modules/orientation/handshape_with_normals.gif'):
    """
    Creates a GIF of the 3D handshapes for both hands and their normal vectors over time.
    
    Parameters:
    - pose_left: numpy array of shape [T, n_nodes, 3] for the left hand.
    - pose_right: numpy array of shape [T, n_nodes, 3] for the right hand.
    - normal_vectors_left: numpy array of shape [T, 3] representing normal vectors for the left hand.
    - normal_vectors_right: numpy array of shape [T, 3] representing normal vectors for the right hand.
    - gif_path: Path to save the output GIF.
    """
    # Angles for viewing the 3D plots
    angles = [[0, 0], [30, -30], [90, 90]]

    # Define edges connecting keypoints for plotting
    inward_edges = [
        [1, 0], [2, 1], [3, 2], [4, 3],     # Thumb
        [5, 0], [6, 5], [7, 6], [8, 7],     # Index Finger
        [9, 0], [10, 9], [11, 10], [12, 11],# Middle Finger
        [13, 0], [14, 13], [15, 14], [16, 15],# Ring Finger
        [17, 0], [18, 17], [19, 18], [20, 19] # Pinky Finger
    ]

    T, n_nodes, _ = pose_right.shape

    # Initialize the figure and axes
    fig = plt.figure(figsize=(16, 8))  # Adjust figure size for 4 columns
    scatters = []
    line_collections = []
    quiver_axes = []

    # Create subplots for both hands
    for row, (pose, normal_vectors) in enumerate(zip([pose_right, pose_left], [normal_vectors_right, normal_vectors_left])):
        for idx_angle, angle in enumerate(angles):
            ax = fig.add_subplot(2, 4, row * 4 + idx_angle + 1, projection='3d')
            ax.set_title(f'{"Right" if row == 0 else "Left"} Hand - View {idx_angle + 1}')
            ax.set_xlim(-0.2, 0.2)
            ax.set_ylim(-0.2, 0.2)
            ax.set_zlim(-0.1, 0.2)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=angle[0], azim=angle[1])

            # Add scatter and lines for this view
            scatter = ax.scatter([], [], [], c='b', s=20)
            lines = [ax.plot([], [], [], 'r-', linewidth=1)[0] for _ in inward_edges]

            scatters.append(scatter)
            line_collections.append(lines)

        # Add subplot for normal vector
        ax_norm = fig.add_subplot(2, 4, row * 4 + 4, projection='3d')
        ax_norm.set_title(f'{"Right" if row == 0 else "Left"} Hand - Normal Vector')
        ax_norm.set_xlim([-1, 1])
        ax_norm.set_ylim([-1, 1])
        ax_norm.set_zlim([-1, 1])
        ax_norm.set_xlabel('X')
        ax_norm.set_ylabel('Y')
        ax_norm.set_zlabel('Z')
        quiver_axes.append(ax_norm)

    # Update function for the animation
    def update(frame):
        # Update hand poses
        for row, (pose, normal_vectors) in enumerate(zip([pose_right, pose_left], [normal_vectors_right, normal_vectors_left])):
            current_keypoint = pose[frame]
            x, y, z = current_keypoint[:, 0], current_keypoint[:, 1], current_keypoint[:, 2]

            for i in range(3):  # 3 angles per hand
                idx = row * 3 + i
                scatters[idx]._offsets3d = (x, y, z)
                for line, (start, end) in zip(line_collections[idx], inward_edges):
                    line.set_data([x[start], x[end]], [y[start], y[end]])
                    line.set_3d_properties([z[start], z[end]])

            # Update normal vector plot
            quiver_axes[row].cla()
            quiver_axes[row].quiver(0, 0, 0, normal_vectors[frame, 0], normal_vectors[frame, 1], normal_vectors[frame, 2],
                                    color='r', length=1, normalize=True)
            quiver_axes[row].set_xlim([-1, 1])
            quiver_axes[row].set_ylim([-1, 1])
            quiver_axes[row].set_zlim([-1, 1])
            quiver_axes[row].set_xlabel('X')
            quiver_axes[row].set_ylabel('Y')
            quiver_axes[row].set_zlabel('Z')
            quiver_axes[row].set_title(f'{"Right" if row == 0 else "Left"} Hand - Normal Vector')

        return scatters + [line for lines in line_collections for line in lines]

    # Create animation
    anim = FuncAnimation(fig, update, frames=T, interval=100, blit=False)

    # Save animation as GIF
    writer = PillowWriter(fps=3)
    anim.save(gif_path, writer=writer)

    plt.close(fig)
    print(f"GIF saved at {gif_path}")
