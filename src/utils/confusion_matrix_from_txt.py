import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

fsize = 18

# Step 1: Load the mapping from the global_value_to_id.txt file into a dictionary
mapping = {}
with open('PoseTools/data/metadata/output/global_value_to_id.txt', 'r') as f:
    for line in f:
        # Each line is in the format 'label: integer'
        key, value = line.strip().split(': ')
        mapping[int(value)] = key  # Map integer to label


def read_predictions_from_file(file_path):
    """
    Reads a text file containing predictions and actual labels and returns two lists: 
    one for predicted labels and one for actual labels.
    """
    predicted_labels = []
    actual_labels = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Predicted"):
                # Extract the predicted and actual labels
                parts = line.strip().split(',')
                predicted = int(parts[0].split(':')[1].strip())
                actual = int(parts[1].split(':')[1].strip())
                
                # Append to the respective lists
                predicted_labels.append(predicted)
                actual_labels.append(actual)
    
    return np.array(predicted_labels), np.array(actual_labels)


# Example usage:
file_path = 'PoseTools/results/logs/predictions_and_labels.txt'
pred, true = read_predictions_from_file(file_path)

# Step 3: Create the confusion matrix
cm = confusion_matrix(true, pred)

# Normalize the confusion matrix
with np.errstate(invalid='ignore', divide='ignore'):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN values with 0

# Convert the normalized values to percentages and round them
cm_percentage = (cm_normalized * 100).astype(int)


label_to_label_map = {0: 'B_bent', 1: 'B_curved', 2: 'S', 3: 'V', 4: 'C', 5: '1_curved', 6: 'B', 7: 'Money', 8: '1', 9: 'V_curved', 10: '5', 11: 'Baby_C', 12: 'C_spread', 13: 'Y', 14: 'N', 15: 'T', 16: 'A', 17: 'L', 18: '5m', 19: 'Beak'}

# 20c_200
#{
#    0: '1_curved', 1: 'N', 2: 'V', 3: 'Money', 4: 'Beak', 5: 'Baby_C', 
#    6: '1', 7: 'T', 8: 'B', 9: 'Y', 10: 'S', 11: 'C', 12: 'B_curved', 
#    13: 'A', 14: 'V_curved', 15: 'L', 16: 'C_spread', 17: 'B_bent', 
#    18: '5m', 19: '5'
#}

# Reverse the key-value mapping
label_to_label_map = {v: k for k, v in label_to_label_map.items()}

#{'B': 0,'C': 1,'A': 2,'5': 3,'1': 4,'Money': 5,'S': 6,'C_spread': 7,'T': 8,'V': 9 } 
#{'C': 0, 'S': 1, 'B': 2, '1': 3, '5': 4}
#{"Y": 0, "B_curved": 1, "N": 2, "S": 3, "Beak": 4, "1_curved": 5, "V": 6, "1": 7, "5": 8, "Baby_C": 9,  "L": 10,  "Money": 11,  "T": 12,  "V_curved": 13,  "C_spread": 14,  "B_bent": 15,  "B": 16,  "A": 17,  "C": 18,  "5m": 19}
# {"1": 0, "T": 1, "V": 2, "A": 3, "C_spread": 4, "B": 5, "S": 6, "C": 7, "Y": 8, "5": 9}
#  
# ToDo find the values in true (1-9) in the label to label map
class_names = [value for value, key in label_to_label_map.items()]
# Step 4: Use the mapping dictionary to replace the integer labels with the corresponding class names
#class_names = [mapping[int(label)] for label in true_labels]  # Get the class names in the correct order

# Step 5: Plot the confusion matrix with the correct labels
plt.figure(figsize=(15, 12))  # Larger figure size
sns.heatmap(cm_percentage, annot=True, fmt='d', cmap='Blues',  # Display as integer percentages with no decimals
            xticklabels=class_names, yticklabels=class_names, 
            annot_kws={"size": fsize})  # Smaller annotation font size

# Rotate tick labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=fsize-2)  # Rotate and adjust font size
plt.yticks(rotation=0, fontsize=fsize-2)

# Set axis labels and title
plt.xlabel('Predicted Class', fontsize=fsize)
plt.ylabel('True Class', fontsize=fsize)
plt.title('Normalized Confusion Matrix for Handshape Classification', fontsize=fsize+2)

# Save the plot to a file
plt.tight_layout()  # Adjust layout to prevent label cut-off
plt.savefig('/home/gomer/oline/PoseTools/results/logs/cm.png')

print("Confusion matrix saved as 'cm.png'")
