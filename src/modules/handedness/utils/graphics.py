import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import numpy.ma as ma
from pose_format import Pose

import matplotlib.pyplot as plt
import imageio
from pose_format import Pose
from PoseTools.data.parsers_and_processors.processors import MediaPipeProcessor
from PoseTools.src.utils.preprocessing import PoseSelect

class PosePlotter:
    def __init__(self, pose = None, path="A.pose"):
        self.pose_path = path
        self.pose = pose
        self.data = None
        self.conf = None
        self.wrist_index_left = 12
        self.wrist_index_right = 13
        self.right_hand_base_index = np.array([16, 18, 20, 22])
        self.left_hand_base_index = np.array([15, 17, 19, 21])
        self.n = len(self.right_hand_base_index)

    def load_pose(self):
        with open(self.pose_path, "rb") as file:
            data_buffer = file.read()

        self.pose = Pose.read(data_buffer)
        data = self.pose.body.data.data
        conf = self.pose.body.confidence

        mp_select = PoseSelect("mediapipe_holistic_minimal_27")
        
        pose = mp_select.clean_keypoints(data)
        pose = mp_select.get_keypoints_pose(pose)

        return pose, conf
        
    
    def plot_mp_skeleton(self, pose, ax=None, mode = "xy"):
        """
        Plot a single frame with keypoints and edges.

        :param pose: Numpy array of shape [T, 27, 2]
        :param frame_idx: The index of the frame to plot
        :param ax: Matplotlib Axes object to plot on. If None, a new figure and axes are created.
        :return: Matplotlib Figure and Axes objects
        """
        x = pose[:, 0]
        y = -pose[:, 1]
        z = pose[:, 2]


        #fig = ax.get_figure()

        # Clear previous plots
        ax.clear()


        # Plot keypoints
        if mode == 'xy':
            ax.scatter(x, y, color='blue')

            right_hand_com_x, right_hand_com_y = 0, 0 
            for ix in self.right_hand_base_index:
                right_hand_com_x += x[ix]
                right_hand_com_y += y[ix]
            #ax.scatter(right_hand_com_x/self.n, right_hand_com_y/self.n, color='red')
            left_hand_com_x, left_hand_com_y = 0, 0
            for ix in self.left_hand_base_index:
                left_hand_com_x += x[ix]
                left_hand_com_y += y[ix]    
            #ax.scatter(left_hand_com_x/self.n, left_hand_com_y/self.n, color='green')
            ax.scatter(x[self.wrist_index_left], y[self.wrist_index_left], color='green')
            ax.scatter(x[self.wrist_index_right], y[self.wrist_index_right], color='red')
            
        elif mode == 'yz':
            
            ax.scatter(z, y, color='blue')
            ax.scatter(z[self.wrist_index_left], y[self.wrist_index_left], color='green')
            ax.scatter(z[self.wrist_index_right], y[self.wrist_index_right], color='red')
        
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1.5)
        # Plot edges
        #for edge in edges:
        #    start, end = edge
        #    ax.plot([x[start], x[end]], [y[start], y[end]], 'r-')

        # Add index numbers to the keypoints
        #for idx, (x_coord, y_coord) in enumerate(zip(x, y)):
        #    ax.text(x_coord, y_coord, str(idx), fontsize=8, color='black', ha='right')

        ax.set_aspect('equal', 'box')
        ax.axis('off')  # Hide axes
        

        #return fig, ax

    def plot_frame_with_edges(self, pose, frame_idx=0, ax=None, mode = "xy"):
        """
        Plot a single frame with keypoints and edges.

        :param pose: Numpy array of shape [T, 27, 2]
        :param frame_idx: The index of the frame to plot
        :param ax: Matplotlib Axes object to plot on. If None, a new figure and axes are created.
        :return: Matplotlib Figure and Axes objects
        """

        frame = pose[frame_idx]  # Shape [27, 2]
        x = frame[:, 0]
        y = frame[:, 1]
        z = frame[:, 2]


        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.get_figure()

        # Clear previous plots
        ax.clear()


        # Plot keypoints
        if mode == 'xy':
            ax.scatter(x, y, color='blue', s = 10)
            right_hand_base_index = [16, 18, 20, 22]
            left_hand_base_index = [15, 17, 19, 21]
            n = len(right_hand_base_index)
            right_hand_com_x, right_hand_com_y = 0, 0 
            for ix in right_hand_base_index:
                right_hand_com_x += x[ix]
                right_hand_com_y += y[ix]
            ax.scatter(right_hand_com_x/n, right_hand_com_y/n, color='red', s = 10)
            left_hand_com_x, left_hand_com_y = 0, 0
            for ix in left_hand_base_index:
                left_hand_com_x += x[ix]
                left_hand_com_y += y[ix]    
            ax.scatter(left_hand_com_x/n, left_hand_com_y/n, color='green', s = 10)
            #ax.scatter(x[self.wrist_index_left], y[self.wrist_index_left], color='green')
            #ax.scatter(x[self.wrist_index_right], y[self.wrist_index_right], color='red')
            
        elif mode == 'yz':
            
            ax.scatter(z, y, color='blue', s = 10)
            ax.scatter(z[self.wrist_index_left], y[self.wrist_index_left], color='green', s = 10)
            ax.scatter(z[self.wrist_index_right], y[self.wrist_index_right], color='red', s = 10)
        
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1.5)
        # Plot edges
        #for edge in edges:
        #    start, end = edge
        #    ax.plot([x[start], x[end]], [y[start], y[end]], 'r-')

        # Add index numbers to the keypoints
        #for idx, (x_coord, y_coord) in enumerate(zip(x, y)):
        #    ax.text(x_coord, y_coord, str(idx), fontsize=8, color='black', ha='right')

        ax.set_aspect('equal', 'box')
        ax.axis('off')  # Hide axes
        

        return fig, ax

    def create_gif(self, pose, gif_path="pose_animation.gif", fps=10):
        """
        Create a GIF from the pose series.

        :param pose: Numpy array of shape [T, 27, 2]
        :param gif_path: Path to save the GIF
        :param fps: Frames per second for the GIF
        """
        temp_dir = "temp_pose_frames"
        os.makedirs(temp_dir, exist_ok=True)
        filenames = []

        print("Generating frames...")
        for frame_idx in range(pose.shape[0]):
            fig, ax = self.plot_frame_with_edges(pose, frame_idx)
            filename = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            filenames.append(filename)
            plt.close(fig)  # Close the figure to free memory

        print("Creating GIF...")
        with imageio.get_writer(gif_path, mode='I', fps=fps) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        print(f"GIF saved to {gif_path}")

        # Optional: Clean up temporary frames
        for filename in filenames:
            os.remove(filename)
        os.rmdir(temp_dir)
        print("Temporary frames cleaned up.")

    

    def create_gif_with_animation(self, pose, gif_path="pose_animation.gif", fps=10):
        """
        Alternative method to create a GIF using Matplotlib's animation.

        :param pose: Numpy array of shape [T, 27, 2]
        :param gif_path: Path to save the GIF
        :param fps: Frames per second for the GIF
        """
        import matplotlib.animation as animation

        fig, ax = plt.subplots(figsize=(6, 6))
        edges = [
            (0, 1), (1, 2), (2, 3),
            (5, 9), (4, 8), (3, 4), (4, 5), (4, 7),
            (4, 6), (6, 10), (9, 13), (14, 18),
            (13, 17), (7, 11), (8, 12), (12, 16),
            (11, 15), (16, 20), (10, 14), (15, 19)
        ]

        scat = ax.scatter([], [], color='blue')
        lines = [ax.plot([], [], 'r-')[0] for _ in edges]
        texts = [ax.text(0, 0, str(idx), fontsize=8, color='black', ha='right') for idx in range(27)]

        ax.set_aspect('equal', 'box')
        ax.axis('off')  # Hide axes

        def init():
            scat.set_offsets([])
            for line in lines:
                line.set_data([], [])
            for text in texts:
                text.set_position((0, 0))
            return [scat] + lines + texts

        def animate(frame_idx):
            frame = pose[frame_idx]
            x = frame[:, 0]
            y = frame[:, 1]
            offsets = np.column_stack((x, y))
            scat.set_offsets(offsets)

            for idx, edge in enumerate(edges):
                start, end = edge
                lines[idx].set_data([x[start], x[end]], [y[start], y[end]])

            for idx, text in enumerate(texts):
                text.set_position((x[idx], y[idx]))
                text.set_text(str(idx))

            return [scat] + lines + texts

        ani = animation.FuncAnimation(fig, animate, frames=pose.shape[0],
                                      init_func=init, blit=True)

        ani.save(gif_path, writer='imagemagick', fps=fps)
        plt.close(fig)
        print(f"GIF saved to {gif_path} using animation.")


def plot_velocity(velocity_r, velocity_l, pose_filename):
    # Calculate the magnitude of the velocity vectors
    vel_r_mag = np.sqrt(np.sum(velocity_r**2, axis=1))  # Use np.sqrt and sum for norm
    vel_l_mag = np.sqrt(np.sum(velocity_l**2, axis=1))

    # Create a 1D plot of the velocity profiles
    frames = np.arange(len(vel_r_mag))

    plt.figure(figsize=(10, 5))
    plt.plot(frames, vel_r_mag, label="Right Hand Velocity")
    plt.plot(frames, vel_l_mag, label="Left Hand Velocity", linestyle='--')

    plt.xlabel("Frame")
    plt.ylabel("Velocity Magnitude")
    plt.title("Velocity Profile of Both Hands")
    plt.legend()
    plt.grid(True)
    plt.savefig('graphics/velocity_'+pose_filename+'.png')


def plot_integrated_velocities(integrated_velocities, output_file):
    # Create bar plot for integrated velocities
    n_files = len(integrated_velocities)
    
    # Separate right and left hand data
    integrated_r = [item[0] for item in integrated_velocities]
    integrated_l = [item[1] for item in integrated_velocities]
    
    # Generate bar plot
    x = np.arange(n_files)  # X-axis values (just indices)
    
    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, integrated_r, width=0.4, label="Right Hand", color="red")
    plt.bar(x + 0.2, integrated_l, width=0.4, label="Left Hand", color="blue")
    
    plt.ylabel("Integrated Velocity")
    plt.title("Integrated Velocity of Both Hands across Files")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(output_file)
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_position_and_velocity(pos_r, pos_l, active_r, active_l, vel_r, vel_l, pose_filename):
    """
    Plot the position and velocity profiles of both hands in a single figure with two subplots.
    
    Parameters:
    - pos_r (list or np.ndarray): Y positions of the right hand.
    - pos_l (list or np.ndarray): Y positions of the left hand.
    - active_r (list or np.ndarray): Boolean flags indicating active frames for the right hand.
    - active_l (list or np.ndarray): Boolean flags indicating active frames for the left hand.
    - vel_r (list or np.ndarray): Velocity magnitudes of the right hand.
    - vel_l (list or np.ndarray): Velocity magnitudes of the left hand.
    - pose_filename (str): Filename to save the plot.
    """
    frames = np.arange(len(pos_r))
    
    # Convert inputs to NumPy arrays for easier indexing
    pos_r = np.array(pos_r)
    pos_l = np.array(pos_l)
    active_r = np.array(active_r)
    active_l = np.array(active_l)
    vel_r = np.array(vel_r)
    vel_l = np.array(vel_l)
    
    # Create a figure with two subplots: position and velocity
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # ---- Top Subplot: Position Plot ----
    # Separate active and inactive frames for both hands
    pos_r_active = pos_r[active_r]
    frames_active_r = frames[active_r]
    
    pos_r_inactive = pos_r[~active_r]
    frames_inactive_r = frames[~active_r]
    
    pos_l_active = pos_l[active_l]
    frames_active_l = frames[active_l]
    
    pos_l_inactive = pos_l[~active_l]
    frames_inactive_l = frames[~active_l]
    
    # Plot inactive points with alpha=0.2
    if len(pos_r_inactive) > 0:
        ax1.scatter(frames_inactive_r, pos_r_inactive, label="Right Hand Inactive", color='blue', alpha=0.2)
    if len(pos_l_inactive) > 0:
        ax1.scatter(frames_inactive_l, pos_l_inactive, label="Left Hand Inactive", color='orange', alpha=0.2)
    
    # Plot active points with alpha=1
    if len(pos_r_active) > 0:
        ax1.scatter(frames_active_r, pos_r_active, label="Right Hand Active", color='blue', alpha=1.0)
    if len(pos_l_active) > 0:
        ax1.scatter(frames_active_l, pos_l_active, label="Left Hand Active", color='orange', alpha=1.0)
    
    ax1.set_ylabel("Position Y Coordinate (Normalized)")
    ax1.set_title("Hand Position Profiles with Activity Indication")
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # ---- Bottom Subplot: Velocity Plot ----
    ax2.plot(frames, vel_r, label="Right Hand Velocity", color='blue')
    ax2.plot(frames, vel_l, label="Left Hand Velocity", color='orange', linestyle='--')
    
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Velocity Magnitude")
    ax2.set_title("Velocity Profiles of Both Hands")
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = f'demo_files/graphics/position_velocity_{pose_filename}.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    print(f"Combined plot saved at {output_path}")

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_position_velocity_product(pos_r, pos_l, active_r, active_l, vel_r, vel_l, pos_vel_r, pos_vel_l, pose_filename):
    """
    Plot the position, velocity, and their product profiles of both hands in a single figure with three subplots.
    
    Parameters:
    - pos_r (list or np.ndarray): Y positions of the right hand.
    - pos_l (list or np.ndarray): Y positions of the left hand.
    - active_r (list or np.ndarray): Boolean flags indicating active frames for the right hand.
    - active_l (list or np.ndarray): Boolean flags indicating active frames for the left hand.
    - vel_r (list or np.ndarray): Normalized velocity magnitudes of the right hand.
    - vel_l (list or np.ndarray): Normalized velocity magnitudes of the left hand.
    - pos_vel_r (list or np.ndarray): Product of position and velocity for the right hand.
    - pos_vel_l (list or np.ndarray): Product of position and velocity for the left hand.
    - pose_filename (str): Filename to save the plot.
    """
    frames = np.arange(len(pos_r))
    
    # Convert inputs to NumPy arrays for easier indexing
    pos_r = np.array(pos_r)
    pos_l = np.array(pos_l)
    active_r = np.array(active_r)
    active_l = np.array(active_l)
    vel_r = np.array(vel_r)
    vel_l = np.array(vel_l)
    pos_vel_r = np.array(pos_vel_r)
    pos_vel_l = np.array(pos_vel_l)
    
    # Create a figure with three subplots: position, velocity, and position*velocity
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
    
    # ---- Top Subplot: Position Plot ----
    # Separate active and inactive frames for both hands
    pos_r_active = pos_r[active_r]
    frames_active_r = frames[active_r]
    
    pos_r_inactive = pos_r[~active_r]
    frames_inactive_r = frames[~active_r]
    
    pos_l_active = pos_l[active_l]
    frames_active_l = frames[active_l]
    
    pos_l_inactive = pos_l[~active_l]
    frames_inactive_l = frames[~active_l]
    
    # Plot inactive points with alpha=0.2
    if len(pos_r_inactive) > 0:
        ax1.scatter(frames_inactive_r, pos_r_inactive, label="Right Hand Inactive", color='blue', alpha=0.2)
    if len(pos_l_inactive) > 0:
        ax1.scatter(frames_inactive_l, pos_l_inactive, label="Left Hand Inactive", color='orange', alpha=0.2)
    
    # Plot active points with alpha=1
    if len(pos_r_active) > 0:
        ax1.scatter(frames_active_r, pos_r_active, label="Right Hand Active", color='blue', alpha=1.0)
    if len(pos_l_active) > 0:
        ax1.scatter(frames_active_l, pos_l_active, label="Left Hand Active", color='orange', alpha=1.0)
    
    ax1.set_ylabel("Position Y Coordinate (Normalized)")
    ax1.set_title("Hand Position Profiles with Activity Indication")
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # ---- Middle Subplot: Velocity Plot ----
    ax2.plot(frames, vel_r, label="Right Hand Velocity (Normalized)", color='blue')
    ax2.plot(frames, vel_l, label="Left Hand Velocity (Normalized)", color='orange', linestyle='--')
    
    ax2.set_ylabel("Velocity Magnitude (Normalized)")
    ax2.set_title("Normalized Velocity Profiles of Both Hands")
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    # ---- Bottom Subplot: Position * Velocity Plot ----
    ax3.plot(frames, pos_vel_r, label="Right Hand Position*Velocity", color='blue')
    ax3.plot(frames, pos_vel_l, label="Left Hand Position*Velocity", color='orange', linestyle='--')
    
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("Position * Velocity (Normalized)")
    ax3.set_title("Normalized Position * Velocity Profiles of Both Hands")
    ax3.legend(loc='upper right')
    ax3.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = f'demo_files/graphics/position_velocity_product_{pose_filename}.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    print(f"Combined plot saved at {output_path}")


def plot_position(pos_r, pos_l, active_r, active_l, pose_filename):
    """
    Plot the position profiles of both hands with varying transparency based on activity.
    
    Parameters:
    - pos_r (list or np.ndarray): Y positions of the right hand.
    - pos_l (list or np.ndarray): Y positions of the left hand.
    - active_r (list or np.ndarray): Boolean flags indicating active frames for the right hand.
    - active_l (list or np.ndarray): Boolean flags indicating active frames for the left hand.
    - pose_filename (str): Filename to save the plot.
    """
    frames = np.arange(len(pos_r))
    
    # Convert inputs to NumPy arrays for easier indexing
    pos_r = np.array(pos_r)
    pos_l = np.array(pos_l)
    active_r = np.array(active_r)
    active_l = np.array(active_l)
    
    # Separate active and inactive frames for both hands
    pos_r_active = pos_r[active_r]
    frames_active_r = frames[active_r]
    
    pos_r_inactive = pos_r[~active_r]
    frames_inactive_r = frames[~active_r]
    
    pos_l_active = pos_l[active_l]
    frames_active_l = frames[active_l]
    
    pos_l_inactive = pos_l[~active_l]
    frames_inactive_l = frames[~active_l]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot inactive points with alpha=0.2
    plt.scatter(frames_inactive_r, pos_r_inactive, label="Right Hand Inactive", color='blue', alpha=0.1)
    plt.scatter(frames_inactive_l, pos_l_inactive, label="Left Hand Inactive", color='orange', alpha=0.1)
    
    # Plot active points with alpha=1
    plt.scatter(frames_active_r, pos_r_active, label="Right Hand Active", color='blue', alpha=1.0)
    plt.scatter(frames_active_l, pos_l_active, label="Left Hand Active", color='orange', alpha=1.0)
    
    plt.xlabel("Frame")
    plt.ylabel("Position Y Coordinate (Normalized)")
    plt.title("Hand Position Profiles with Activity Indication")
    plt.legend()
    plt.grid(True)
    
    # Ensure the directory exists
    output_path = f'demo_files/graphics/position_{pose_filename}.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path)
    plt.close()
    
    print(f"Plot saved at {output_path}")


def plot_hamer_hand_3d(node_positions, output_file_name):
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
    
    # Create a 2x2 grid for subplots
    fig = plt.figure(figsize=(12, 12))
    angles = [[0,0],[5, 45],[45, 0],[45, 45]]

    fig.suptitle(f"3D Handshape Visualization for Gloss: {output_file_name}", fontsize=16)
    
    for i, angle in enumerate(angles):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        
        # Plot the nodes
        ax.scatter(node_positions[:, 0], node_positions[:, 1], node_positions[:, 2], color='b', s=100)
        
        # Plot the edges (connections)
        for connection in connections:
            start, end = connection
            xs = [node_positions[start, 0], node_positions[end, 0]]
            ys = [node_positions[start, 1], node_positions[end, 1]]
            zs = [node_positions[start, 2], node_positions[end, 2]]
            ax.plot(xs, ys, zs, color='r', linewidth=2)

        # Set labels and viewing angle
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=angle[0], azim=angle[1])
        
        ax.set_title(f"View angle: {angle}°")

    # Save the plot as a single image
    plt.tight_layout()
    plt.savefig(f'PoseTools/src/modules/handshapes/utils/reference_poses/{output_file_name}.png')
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

def read_dict_from_txt(filename):
    """
    Reads a dictionary from a .txt file where each line is in the format 'key: value'.
    
    Parameters:
    - filename: The name of the input file.
    
    Returns:
    - A dictionary with the key-value pairs from the file.
    """
    value_to_id = {}
    with open(filename, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            value_to_id[int(value)] = key  # Store with value as the key and key as the value
    return value_to_id


def calculate_euclidean_distance(pose, reference_poses = None, gloss_mapping = None ):
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
    for key, reference_pose in reference_poses.items():
        distances.append(np.linalg.norm(pose - reference_pose, axis=1).mean())
        keys.append(key)
    closest_handshape = gloss_mapping[int(keys[np.argmin(np.array(distances))])]
    return closest_handshape

def calculate_top_n_closest_handshapes(pose, reference_poses = None, gloss_mapping = None,  n=3):
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
        

def plot_multiple_hands_from_dict(node_positions_dict, output_path):
    """
    Plots multiple 3D hand graphs in a 5x7 grid using a dictionary of glosses and node positions.
    
    Parameters:
    - node_positions_dict: A dictionary where keys are integers (1-35) and values are numpy arrays of shape (21, 3).
    - gloss_mapping: A dictionary that maps integers (1-35) to glosses (strings).
    """
    gloss_mapping = read_dict_from_txt('/home/gomer/oline/PoseTools/data/metadata/output/global_value_to_id.txt')

    # Create a figure with a 5x7 grid of subplots
    fig = plt.figure(figsize=(15, 10))

    
    # Loop through each hand's node positions and gloss
    for i, (key, node_positions) in enumerate(node_positions_dict.items()):
        if i > 35: break
        print(i, key)
        print(len(node_positions_dict.keys()))
        ax = fig.add_subplot(5, 7, i+1, projection='3d')

        # Map the numeric ID to the corresponding gloss from gloss_mapping
        gloss = key # gloss_mapping.get(int(key), "Unknown Gloss")
        # Debugging: Print the gloss being used
        
        plot_hand_3d(ax, node_positions, gloss)
        
    
    # Adjust layout to avoid overlap
    plt.tight_layout()
    
    plt.savefig(output_path)