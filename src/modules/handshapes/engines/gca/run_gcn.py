from PoseTools.src.models.graphTransformer.gt import HandshapeGAT, train_model, test_model
from torch.optim.lr_scheduler import CyclicLR

from PoseTools.src.models.graphTransformer.utils.dataloader import GraphDataReader, GraphDataLoader
import argparse
import os
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import LambdaLR, ExponentialLR

def main(args):
    device = args.device
    
    # Load Data
    data = GraphDataReader(args)
    
    # Initialize Dataloader
    dataloader = GraphDataLoader(data, args)
    
    # Create train, validation, and test loaders
    train_loader = dataloader.train_loader
    val_loader = dataloader.val_loader
    test_loader = dataloader.test_loader

    # Initialize the model
    model = HandshapeGAT(in_channels=3, hidden_channels=128, out_channels=args.n_classes, heads=8)
    
    # Optimizer and Loss Function
    weight_decay = 1e-5  # You can tune this value
    learning_rate = 1e-3  # Initial learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning Rate Scheduler: Adjust learning rate after every 10 epochs
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Reduce LR by factor of 0.1 every 10 epochs
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)  # Decay LR by 10% after each epoch
    # Define the warmup scheduler
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return 1.0 # Linearly increase the LR
        return 1.0  # Keep the LR at the base value after warmup
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Define another scheduler for after warmup (e.g., ExponentialLR)
    main_scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    #torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    #CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-3, step_size_up=5, mode='triangular2')


    #class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.2, 1.3, 1.3, 1.3, 1.7, 1.75]).to(device)  # Adjust based on counts
    #criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss()
    # Train the model and log to WandB, save best model
    save_dir = os.path.abspath(args.save_dir)  # Directory to save best model
    train_model(model, train_loader, val_loader, optimizer, criterion, device, args.epochs, save_dir, warmup_scheduler=warmup_scheduler, main_scheduler=main_scheduler, test_loader=test_loader, eval = True)

    # Test the model after training
    #test_model(model, test_loader, criterion, device, '/home/gomer/oline/PoseTools/src/models/graphTransformer/trained_models/10c_300_acc_78.05.pth')
    #test_model(model, test_loader, criterion, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    path = "/home/gomer/oline/PoseTools/data/metadata/output/20c/20c_200.json"
    print(path)
    parser.add_argument('--n_classes', type=int,  default= 20,
                        help='Number of handshape classes')
    parser.add_argument('--save_dir', type=str,  default=os.path.abspath("/home/gomer/oline/PoseTools/src/models/graphTransformer/trained_models"),
                        help='Metadata json file location')
    parser.add_argument('--root_metadata', type=str,  default=os.path.abspath(path),
                        help='Metadata json file location')
    parser.add_argument('--root_poses', type=str, default=os.path.abspath("../../../../mnt/fishbowl/gomer/oline/hamer_cleaned"),
                        help='Pose data dir location')
    parser.add_argument('--n_nodes', type=int, default=21,
                        help='Number of nodes to use when reducing the graph - only 27 currently implemented')

    parser.add_argument('--top_n', type=int, default=1,
                        help='Number of handshapes in top-n similar shapes for euclidean preprocessing')


    # Run parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cpu or cuda)')
    
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of epochs for learning rate warmup')


    args = parser.parse_args()
    main(args)
