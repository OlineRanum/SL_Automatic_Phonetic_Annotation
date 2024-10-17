import torch
import argparse
import pickle
import os
from torch_geometric.data import Data
from PoseTools.src.models.graphTransformer.gt import HandshapeGAT
import matplotlib.pyplot as plt
import numpy as np
import imageio
from mpl_toolkits.mplot3d import Axes3D

# Define the handshape labels (as an example)
handshape_labels = ["T", "B", "1", "C", "S"]

def load_model(model_path, device):
    """
    Load the pretrained model from the specified path.
    """
    model = HandshapeGAT(in_channels=3, hidden_channels=128, out_channels=5, heads=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_pose(pose):
    """
    Convert the input pose data into the format expected by the model.
    Create a torch_geometric Data object containing the pose and edge information.
    """
    # Create the edge_index (the connectivity between keypoints). Ensure that this matches the expected structure.
    inward_edges = [
        [1, 0], [2, 1], [3, 2], [4, 3],  # Thumb
        [5, 0], [6, 5], [7, 6], [8, 7],  # Index Finger
        [9, 0], [10, 9], [11, 10], [12, 11],  # Middle Finger
        [13, 0], [14, 13], [15, 14], [16, 15],  # Ring Finger
        [17, 0], [18, 17], [19, 18], [20, 19]  # Pinky Finger
    ]
    edge_index_tensor = torch.tensor(inward_edges).t().contiguous()  # Spatial edges connecting keypoints

    # List to hold Data objects for each frame
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
        data = Data(x=frame_tensor, edge_index=edge_index_tensor)  # Create Data object
        data_list.append(data)

    return data_list

def predict_and_plot_handshape(model, data_list, device, keypoints, output_gif_path, filename):
    """
    Perform inference to predict the handshape label from the list of pose frames.
    Averages predictions across all frames and creates a GIF with predicted labels as titles.
    """
    model.eval()
    predictions = []
    frames = []  # To store frames for GIF
    inward_edges = [
        [1, 0], [2, 1], [3, 2], [4, 3],  # Thumb
        [5, 0], [6, 5], [7, 6], [8, 7],  # Index Finger
        [9, 0], [10, 9], [11, 10], [12, 11],  # Middle Finger
        [13, 0], [14, 13], [15, 14], [16, 15],  # Ring Finger
        [17, 0], [18, 17], [19, 18], [20, 19]  # Pinky Finger
    ]
    
    with torch.no_grad():  # Disable gradient calculation for inference
        for i, data in enumerate(data_list):
            data = data.to(device)  # Move each Data object to the device
            output = model(data)  # Perform inference on each frame
            _, predicted = torch.max(output, dim=1)  # Get the predicted class for this frame
            predictions.append(predicted.item())

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
            ax.set_title(f'Ground truth {filename}\nPredicted Handshape: {handshape_labels[predicted.item()]}')

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

def process_directory(input_folder, model, device):
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
            output_gif_path = os.path.join(input_folder, f"gif/{os.path.splitext(filename)[0]}_output.gif")
            predicted_class_idx = predict_and_plot_handshape(model, pose_data, device, pose, output_gif_path, filename[:-4])

            # Convert the predicted index to a human-readable handshape label
            predicted_label = handshape_labels[predicted_class_idx]
            print(f"File: {filename} - Predicted Handshape: {predicted_label}")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the pretrained model
    model = load_model(args.model_path, device)

    # Process all .pkl files in the specified directory
    process_directory(args.input_folder, model, device)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default='/home/gomer/oline/PoseTools/src/models/graphTransformer/test_data/vids/norm/pkl/', 
                        type=str, help='Directory containing .pkl pose data files')
    parser.add_argument('--model_path', default='/home/gomer/oline/PoseTools/src/models/graphTransformer/trained_models/5c_cleaned_98.pth',
                        type=str, help='Path to the saved model weights')

    args = parser.parse_args()
    main(args)
