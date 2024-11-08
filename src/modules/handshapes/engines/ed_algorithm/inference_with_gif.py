import torch
import argparse
import pickle
import os
from torch_geometric.data import Data
from PoseTools.src.models.gca.gt import HandshapeGAT
import matplotlib.pyplot as plt
import numpy as np
import imageio
from mpl_toolkits.mplot3d import Axes3D
from PoseTools.src.modules.handedness.utils.graphics import read_dict_from_txt


with open('/home/gomer/oline/PoseTools/src/modules/handshapes/utils/references/reference_poses.pkl', 'rb') as file:
    reference_poses = pickle.load(file)

# List of keys to keep
keys_to_keep = ['1', '2', '3', '4', '9', '12', '17', '18','19', '22', '23', '25', '27', '36','37','38',  '41', '42', '43', '44']
#['2', '3', '4', '17', '18']

# Create a new dictionary with only the specified keys
reference_poses = {key: reference_poses[key] for key in reference_poses}

gloss_mapping = read_dict_from_txt('/home/gomer/oline/PoseTools/data/metadata/output/global_value_to_id.txt')


def calculate_euclidean_distance(pose):
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

def preprocess_pose(pose):
    """
    Convert the input pose data into the format expected by the model.
    Create a torch_geometric Data object containing the pose and edge information.
    """
    data_list = []
    import numpy as np
    new_position = np.array([0.022144492431246672, 0.07874330869633958, 1.1457862878850556e-18])

    # Get the current position of node 0 for each frame
    current_position = pose[:, 0, :]  # Shape [frames, dims]

    # Compute the shift vector: difference between new position and current position of node 0
    shift_vector = new_position - current_position  # Shape [frames, dims]

    # Apply the shift to all keypoints in all frames
    pose = pose + shift_vector[:, np.newaxis, :]  # Broadcasting shift across all keypoints

    # Iterate over each frame in the pose data
    for frame in pose:
        frame_tensor = torch.tensor(frame, dtype=torch.float32)  # Convert frame to tensor
        data_list.append(frame_tensor)
    return data_list


def predict_and_plot_handshape(data_list, keypoints, output_gif_path, filename):
    """
    Perform inference to predict the handshape label from the list of pose frames.
    Averages predictions across all frames and creates a GIF with predicted labels as titles.
    """
    predictions = []
    frames = []  # To store frames for GIF
    inward_edges = [
        [1, 0], [2, 1], [3, 2], [4, 3],  # Thumb
        [5, 0], [6, 5], [7, 6], [8, 7],  # Index Finger
        [9, 0], [10, 9], [11, 10], [12, 11],  # Middle Finger
        [13, 0], [14, 13], [15, 14], [16, 15],  # Ring Finger
        [17, 0], [18, 17], [19, 18], [20, 19]  # Pinky Finger
    ]
    
    for i, data in enumerate(data_list):
        h_label = calculate_euclidean_distance(data)
        
        predictions.append(h_label)

        # Plot the current frame with the predicted label as title
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        x = keypoints[i, :, 0]
        y = keypoints[i, :, 1]
        z = keypoints[i, :, 2]
        
        ax.scatter(x, y, z, c='b', s=20)  # Blue points for keypoints

        # Plot edges
        for edge in inward_edges:
            start = edge[0]
            end = edge[1]
            ax.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], 'r-')  # Red lines for edges

        # Set plot limits and labels
        ax.set_xlim([-0.1, 0.1])
        ax.set_ylim([-0.1, 0.1])
        ax.set_zlim([-0.1, 0.1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set the title as the predicted label
        ax.set_title(f'Ground truth {filename}\nPredicted Handshape: {h_label}')

        # Save the plot as a frame
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

        plt.close(fig)

    # Save the frames as an animated GIF
    imageio.mimsave(output_gif_path, frames, fps=10)
    
    # Majority vote or average over frames (optional)
    most_common_prediction = max(set(predictions), key=predictions.count)
    return most_common_prediction

def process_directory(input_folder, output_folder,  device):
    """
    Process all files in the directory, perform inference on each, and print or save the predicted handshape label.
    """
    # Loop through all .pkl files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.pkl'):  # Only process .pkl files
            file_path = os.path.join(input_folder, filename)
            print(f"Processing file: {file_path}")

            # Load the pose data from the .pkl file
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            # Access the keypoints data
            pose = data['keypoints']  # Ensure this is the correct way to access keypoints

            # Preprocess the pose into a torch_geometric Data object
            pose_data = preprocess_pose(pose)

            # Perform inference and create GIF
            output_gif_path = os.path.join(output_folder,  f"{os.path.splitext(filename)[0]}_output.gif")
            
            predicted_class = predict_and_plot_handshape( pose_data, pose, output_gif_path, filename[:-4])

            # Convert the predicted index to a human-readable handshape label
            #predicted_label = handshape_labels[predicted_class_idx]
            print(f"File: {filename} - Predicted Handshape: {predicted_class}")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    # Process all .pkl files in the specified directory
    process_directory(args.input_folder, args.output_folder,  device)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default='/home/gomer/oline/PoseTools/src/models/graphTransformer/test_data/vids/norm/pkl/', 
                        type=str, help='Directory containing .pkl pose data files')
    parser.add_argument('--output_folder', default='/home/gomer/oline/PoseTools/data/datasets/test_data/gifs/', 
                        type=str, help='Directory containing .pkl pose data files')

    args = parser.parse_args()
    main(args)
