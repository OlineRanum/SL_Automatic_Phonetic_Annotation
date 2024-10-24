import torch
import argparse
import pickle
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Data
from PoseTools.src.models.graphTransformer.gt_tsne import HandshapeGAT

# Define the handshape labels (as an example)
handshape_labels = ["T", "B", "1", "C", "S"]

def load_model(model_path, device):
    """
    Load the pretrained model from the specified path.
    """
    model = HandshapeGAT(in_channels=3, hidden_channels=128, out_channels=10, heads=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_pose(pose):
    """
    Convert the input pose data into the format expected by the model.
    Create a torch_geometric Data object containing the pose and edge information.
    """
    # Create a pose tensor (node features) and convert it to a torch_geometric Data object
    pose_tensor = torch.tensor(pose, dtype=torch.float32)

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
    

def predict_handshape(model, data_list, device):
    model.eval()
    embeddings_list = []
    predictions = []

    with torch.no_grad():
        for data in data_list:
            data = data.to(device)
            output, embeddings = model(data)  # Get both predictions and embeddings
            embeddings_list.append(embeddings.cpu().numpy())  # Collect embeddings
            
            _, predicted = torch.max(output, dim=1)  # Get predicted class
            predictions.append(predicted.item())

    return predictions, embeddings_list  # Return both predictions and embeddings




def process_directory(input_folder, model, device):
    all_embeddings = []
    all_labels = []

    # Process all .pkl files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.pkl'):
            file_path = os.path.join(input_folder, filename)
            print(f"Processing file: {file_path}")

            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            pose = data['keypoints']
            pose_data = preprocess_pose(pose)

            # Perform inference and get embeddings
            predictions, embeddings_list = predict_handshape(model, pose_data, device)
            all_embeddings.extend(embeddings_list)  # Add all embeddings
            all_labels.extend(predictions)  # Add predicted labels

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(np.array(all_embeddings))

    # Plot t-SNE result
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=all_labels, cmap='viridis', s=50)
    plt.colorbar(scatter, ticks=range(len(handshape_labels)), label='Handshape Label')
    plt.xticks([])
    plt.yticks([])
    plt.title("t-SNE Visualization of Handshape Embeddings")
    plt.show()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the pretrained model
    model = load_model(args.model_path, device)

    # Process all .pkl files and visualize t-SNE
    process_directory(args.input_folder, model, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default='/home/gomer/oline/PoseTools/src/models/graphTransformer/test_data/vids/norm/pkl/', 
                        type=str, help='Directory containing .pkl pose data files')
    parser.add_argument('--model_path', default='/home/gomer/oline/PoseTools/src/models/graphTransformer/trained_models/10c_900_acc_90.pth',
                        type=str, help='Path to the saved model weights')

    args = parser.parse_args()
    main(args)