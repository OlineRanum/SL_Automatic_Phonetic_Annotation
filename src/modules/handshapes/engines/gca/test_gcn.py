from PoseTools.src.models.graphTransformer.gt import HandshapeGAT, train_model, test_model
from torch.optim.lr_scheduler import CyclicLR

from PoseTools.src.models.graphTransformer.utils.dataloader import GraphDataReader, GraphDataLoader
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR


label_to_label_map = {
    'B': 0,
    'C': 1,
    'A': 2,
    '5': 3,
    '1': 4,
    'Money': 5,
    'S': 6,
    'C_spread': 7,
    'T': 8,
    'V': 9 } 
'''
# Your label map
label_to_label_map = { 
    'Money': 0,
    'C': 1,
    '5': 2,
    '1': 3,
    'V': 4,
    'B': 5,
    'S': 6,
    'A': 7,
    'C_spread': 8,
    'T' : 9}
    # {'C': 0, 'S': 1, 'B': 2, '1': 3, '5': 4}
'''
# Invert the label map to get numeric to name mapping
numeric_to_label_map = {v: k for k, v in label_to_label_map.items()}

def main(args):
    device = args.device
    
    # Load Data
    data = GraphDataReader(args, numeric_to_label_map)
    
    # Initialize Dataloader
    dataloader = GraphDataLoader(data, args)
    
    # Create train, validation, and test loaders
    test_loader = dataloader.test_loader

    # Initialize the model
    model = HandshapeGAT(in_channels=3, hidden_channels=128, out_channels=args.n_classes, heads=8)
    
    criterion = nn.CrossEntropyLoss()

    # Test the model and get embeddings and labels
    #embeddings, labels = test_model(model, test_loader, criterion, device, args.model_path)
    labels = test_model(model, test_loader, criterion, device, args.model_path)
    
    # Apply t-SNE
    #tsne = TSNE(n_components=2, random_state=42)
    #reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Convert numeric labels to actual label names
    #label_names = [numeric_to_label_map[label] for label in labels]
    
    # Plot t-SNE result
    #plt.figure(figsize=(10, 7))
    #scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', s=50)
    
    # Create colorbar with actual handshape class names
    #cbar = plt.colorbar(scatter, ticks=range(args.n_classes))
    #cbar.set_ticklabels([numeric_to_label_map[i] for i in range(args.n_classes)])
    #cbar.set_label('Handshape Class')

    # plt.xticks([])
    #plt.yticks([])
    #plt.title("t-SNE Visualization of Handshape Embeddings")
    
    # Save the plot
    #plt.savefig("/home/gomer/oline/PoseTools/results/logs/tsne.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    path =  "/home/gomer/oline/PoseTools/data/metadata/output/10c/10c_SB.json"
    model_path =     '/home/gomer/oline/PoseTools/src/models/graphTransformer/trained_models/best_model_epoch_8_val_acc_85.98.pth'
    print(path)
    parser.add_argument('--n_classes', type=int,  default= 10,
                        help='Number of handshape classes')
    parser.add_argument('--top_n', type=int,  default= 10,
                        help='Number of handshape classes')
    parser.add_argument('--save_dir', type=str,  default=os.path.abspath("/home/gomer/oline/PoseTools/src/models/graphTransformer/trained_models"),
                        help='Metadata json file location')
    parser.add_argument('--root_metadata', type=str,  default=os.path.abspath(path),
                        help='Metadata json file location')
    parser.add_argument('--root_poses', type=str, default=os.path.abspath("../../../../mnt/fishbowl/gomer/oline/hamer_pkl"),
                        help='Pose data dir location')
    parser.add_argument('--n_nodes', type=int, default=21,
                        help='Number of nodes to use when reducing the graph - only 27 currently implemented')

    # Run parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cpu or cuda)')
    
    parser.add_argument('--model_path', type=str,  default=os.path.abspath(model_path),
                        help='Path to model')

    args = parser.parse_args()
    main(args)
