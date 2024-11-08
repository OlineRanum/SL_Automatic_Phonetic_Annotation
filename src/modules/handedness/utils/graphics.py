import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

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

def plot_position(pos_r, pos_l, pose_filename):
    # Calculate the magnitude of the velocity vectors

    # Create a 1D plot of the velocity profiles
    frames = np.arange(len(pos_r))

    plt.figure(figsize=(10, 5))
    plt.scatter(frames, pos_r, label="Right Hand Velocity")
    plt.scatter(frames, pos_l, label="Left Hand Velocity", linestyle='--')

    plt.xlabel("Frame")
    plt.ylabel("Position Magnitude")
    plt.title("Position Profile of Both Hands")
    plt.legend()
    plt.grid(True)
    plt.savefig('graphics/position_'+pose_filename+'.png')
    plt.close()


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