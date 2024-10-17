import torch.nn as nn
import torch
import wandb
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import os
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import Set2Set

class HandshapeGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_p=0.3, heads=4):
        super(HandshapeGAT, self).__init__()

        # First GCN layer
        self.gcn1 = GCNConv(in_channels, hidden_channels)

        # Second layer with GAT
        self.gat1 = GATConv(hidden_channels, hidden_channels, heads=heads, concat=True)

        # Batch normalization layers
        self.batch_norm1 = nn.BatchNorm1d(hidden_channels)
        self.batch_norm2 = nn.BatchNorm1d(hidden_channels * heads)

        # Dropout
        self.dropout = nn.Dropout(p=dropout_p)

        # Fully connected layer for classification
        self.fc1 = nn.Linear(hidden_channels * heads, 128)
        self.fc2 = nn.Linear(128, out_channels)

        # Pooling method (global mean pooling)
        self.readout = global_mean_pool

        # Activation function
        self.activation = nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First GCN layer
        x = self.gcn1(x, edge_index)
        x = self.activation(self.batch_norm1(x))
        x = self.dropout(x)

        # Second GAT layer
        x = self.gat1(x, edge_index)
        x = self.activation(self.batch_norm2(x))
        x = self.dropout(x)

        # Graph readout (global mean pooling)
        x = self.readout(x, batch)

        # Fully connected layers
        x = self.fc1(x)
        x = self.activation(x)
        out = self.fc2(x)

        return out


def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs, save_dir, warmup_scheduler, main_scheduler,test_loader = None, eval = True):
    """
    Handles the full training and validation process, including WandB logging and model saving.
    """
    wandb.init(project="trash", config={"epochs": epochs, "batch_size": train_loader.batch_size})
    model.to(device)
    
    best_model_info = {"path": None, "val_acc": 0.0}  # Dictionary to store the best model's info

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        
        # Training loop with tqdm progress bar
        train_progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{epochs}', unit='batch')
        for batch in train_progress_bar:
            optimizer.zero_grad()

            batch = batch.to(device)  # Move batch to GPU/CPU
            labels = batch.y.to(device)  # Labels for classification

            output = model(batch)  # Forward pass
            loss = criterion(output, labels)  # Compute loss

            loss.backward()  # Backpropagation
            optimizer.step()

            total_train_loss += loss.item()

            # Compute training accuracy
            _, predicted = torch.max(output, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

            train_progress_bar.set_postfix(loss=total_train_loss / (train_progress_bar.n + 1))

        train_acc = 100 * correct_train / total_train
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        # Log metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": total_train_loss / len(train_loader),
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']  # Log current learning rate

        })

        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {total_train_loss:.4f}, Training Accuracy: {train_acc:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')

        # Save the model if validation accuracy improves
        if val_acc > best_model_info["val_acc"]:
            model_path = save_best_model(model, save_dir, epoch, val_acc, best_model_info)

        
        # Update the learning rate scheduler
        #if epoch < 5:
        #    warmup_scheduler.step()  # Step warmup scheduler
        #else:
        #    main_scheduler.step()  # Step the main scheduler after warmup


    if eval:
        # Evaluate on the test set before finishing WandB session
        test_acc = test_model(model, test_loader, criterion, device, model_path)
        wandb.log({
            "test_accuracy": test_acc,
            })
    print('Training complete.')
    print('Training complete.')
    wandb.finish()


def evaluate_model(model, loader, criterion, device, save_predictions=False, pred_file_path=None):
    """
    Handles the evaluation (validation/test) process and returns both loss and accuracy.
    If save_predictions is True, it saves the predictions and labels to a file.
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # Lists to store predictions and labels for saving
    all_predictions = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch in loader:
            batch = batch.to(device)
            labels = batch.y.to(device)

            output = model(batch)
            loss = criterion(output, labels)

            total_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(output, 1)  # Get the index of the max logit (predicted class)
            correct += (predicted == labels).sum().item()  # Count correct predictions
            total += labels.size(0)  # Total number of instances

            # Append predictions and labels to lists
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total  # Compute accuracy in percentage

    # Optionally save the predictions and labels
    if save_predictions and pred_file_path:
        with open(pred_file_path, 'w') as f:
            for pred, label in zip(all_predictions, all_labels):
                f.write(f"Predicted: {pred}, Actual: {label}\n")
        print(f"Predictions and labels saved to {pred_file_path}")

    return avg_loss, accuracy


def test_model(model, test_loader, criterion, device, best_model_path):
    """
    Runs the model on the test set, prints the test accuracy and loss, and logs them to WandB.
    Saves the predictions and labels to a file for confusion matrix calculation.
    """
    # Load the best saved model
    print(f'Loading best model from {best_model_path}...')
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Specify where to save the predictions and labels
    pred_file_path = 'predictions_and_labels.txt'

    print('Evaluating on the test set...')
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device, save_predictions=True, pred_file_path=pred_file_path)
    
    # Log test results to WandB
    #wandb.log({
    #    "test_loss": test_loss,
    #    "test_accuracy": test_acc
    #})
    
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    print(f"Predictions and labels saved to {pred_file_path}")
    return test_acc

def save_best_model(model, save_dir, epoch, val_acc, best_model_info):
    """
    Saves the model with the highest validation accuracy and deletes the previous model.
    
    Args:
        model: The model to be saved.
        save_dir: Directory to save the model.
        epoch: The current epoch number.
        val_acc: The current validation accuracy.
        best_model_info: A dictionary containing the last best model's info (path and accuracy).
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create the new model's file path
    model_path = os.path.join(save_dir, f"best_model_epoch_{epoch+1}_val_acc_{val_acc:.2f}.pth")
    
    # If there is a previous model, delete it
    if best_model_info.get("path"):
        previous_model_path = best_model_info["path"]
        if os.path.exists(previous_model_path):
            os.remove(previous_model_path)
    
    # Save the new best model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")
    
    # Update the best model info with the new model's path and accuracy
    best_model_info["path"] = model_path
    best_model_info["val_acc"] = val_acc
    return model_path


'''
class HandshapeGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout_p=0.3):
        super(HandshapeGAT, self).__init__()

        # GATConv layers with more heads and deeper layers
        self.gat1 = GATConv(in_channels, hidden_channels, heads=16, concat=True)
        self.gat2 = GATConv(hidden_channels * 16, hidden_channels, heads=8, concat=True)
        self.gat3 = GATConv(hidden_channels * 8, hidden_channels, heads=4, concat=True)
        self.gat4 = GATConv(hidden_channels * 4, hidden_channels, heads=2, concat=True)  # Extra layer

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_channels * 16)
        self.layer_norm2 = nn.LayerNorm(hidden_channels * 8)
        self.layer_norm3 = nn.LayerNorm(hidden_channels * 4)
        self.layer_norm4 = nn.LayerNorm(hidden_channels * 2)

        # Dropout
        self.dropout = nn.Dropout(p=dropout_p)

        # Fully connected layer for classification
        self.fc1 = nn.Linear(4 * hidden_channels, 128)
        self.fc2 = nn.Linear(128, out_channels)

        # Pooling method (Set2Set pooling)
        self.readout = Set2Set(hidden_channels * 2, processing_steps=3)  # Adjust to hidden_channels * 2

        # Activation function
        self.activation = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GAT layers
        x = self.gat1(x, edge_index)
        x = self.activation(self.layer_norm1(x))
        x = self.dropout(x)

        x = self.gat2(x, edge_index)
        x = self.activation(self.layer_norm2(x))
        x = self.dropout(x)

        x = self.gat3(x, edge_index)
        x = self.activation(self.layer_norm3(x))
        x = self.dropout(x)

        x = self.gat4(x, edge_index)
        x = self.activation(self.layer_norm4(x))
        x = self.dropout(x)

        # Graph readout (Set2Set pooling)
        x = self.readout(x, batch)

        # Fully connected layers
        x = self.fc1(x)
        x = self.activation(x)
        out = self.fc2(x)

        return out
    '''