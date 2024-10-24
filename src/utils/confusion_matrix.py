import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Step 1: Load the mapping from the global_value_to_id.txt file into a dictionary
mapping = {}
with open('/home/gomer/oline/PoseTools/data/metadata/output/global_value_to_id.txt', 'r') as f:
    for line in f:
        # Each line is in the format 'label: integer'
        key, value = line.strip().split(': ')
        mapping[int(value)] = key  # Map integer to label

# Step 2: Load the CSV file containing true and pred values
df = pd.read_csv('/home/gomer/oline/PoseTools/results/logs/output.csv', sep=',')
pred = df['pred'].astype(int)  # Convert to int for consistency
true = df['true'].astype(int)

# Step 3: Create the confusion matrix
cm = confusion_matrix(true, pred)

# Normalize the confusion matrix
with np.errstate(invalid='ignore', divide='ignore'):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN values with 0

# Convert the normalized values to percentages and round them
cm_percentage = (cm_normalized * 100).astype(int)

# Step 4: Use the mapping dictionary to replace the integer labels with the corresponding class names
class_names = [mapping[label] for label in sorted(true.unique())]  # Get the class names in the correct order

# Step 5: Plot the confusion matrix with the correct labels
plt.figure(figsize=(15, 12))  # Larger figure size
sns.heatmap(cm_percentage, annot=True, fmt='d', cmap='Blues',  # Display as integer percentages with no decimals
            xticklabels=class_names, yticklabels=class_names, 
            annot_kws={"size": 12})  # Smaller annotation font size

# Rotate tick labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate and adjust font size
plt.yticks(rotation=0, fontsize=10)

# Set axis labels and title
plt.xlabel('Predicted Class', fontsize=18)
plt.ylabel('True Class', fontsize=18)
plt.title('Normalized Confusion Matrix for Handshape Classification', fontsize=14)

# Save the plot to a file
plt.tight_layout()  # Adjust layout to prevent label cut-off
plt.savefig('/home/gomer/oline/PoseTools/results/logs/cm.png')

print("Confusion matrix saved as 'cm.png'")
