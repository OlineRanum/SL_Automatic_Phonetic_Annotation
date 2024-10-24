import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Load the first text file into a DataFrame
df_txt = pd.read_csv('/home/gomer/oline/PoseTools/src/modules/handedness/euclidean_model/results_euclidean.txt', sep=',', comment='#', header=None,
                     names=['filename', 'handedness', 'gloss', 'handshape'])

# Load the second CSV file into a DataFrame
df_csv = pd.read_csv('/home/gomer/oline/PoseTools/src/modules/handedness/euclidean_model/ground_truth_handshape.csv', sep=',')

# Merge the two DataFrames on the 'filename' column, keeping only rows that exist in both
df_merged = pd.merge(df_txt, df_csv, on='filename', how='inner')

# Strip whitespace from all string columns to ensure proper matching
df_merged = df_merged.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
print(df_merged)
pred = df_merged['handshape']
true = df_merged['strong_hand']
potential_true = df_merged['weak_hand']

# Adjust the true labels to account for matches with potential_true
adjusted_true = np.where(pred == potential_true, potential_true, true)

# Create the confusion matrix using the adjusted true labels
cm = confusion_matrix(adjusted_true, pred)
# Normalize the confusion matrix by dividing each row by the sum of the row (total true instances per class)
# Avoid division by zero when normalizing the confusion matrix
with np.errstate(invalid='ignore', divide='ignore'):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN values with 0

# Convert the normalized values to percentages and round them
cm_percentage = (cm_normalized * 100).astype(int)

# Increase figure size and adjust font size for readability
plt.figure(figsize=(15, 12))  # Larger figure size
sns.heatmap(cm_percentage, annot=True, fmt='d', cmap='Blues',  # Display as integer percentages with no decimals
            xticklabels=sorted(true.unique()), yticklabels=sorted(true.unique()), 
            annot_kws={"size": 8})  # Smaller annotation font size
# Rotate tick labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate and adjust font size
plt.yticks(rotation=0, fontsize=10)

# Set axis labels and title
plt.xlabel('Predicted Handshape', fontsize=12)
plt.ylabel('True Handshape', fontsize=12)
plt.title('Normalized Confusion Matrix for Handshape Classification', fontsize=14)

# Show plot
plt.tight_layout()  # Adjust layout to prevent label cut-off
plt.savefig('/home/gomer/oline/PoseTools/src/modules/handedness/euclidean_model/confusion_matrix.png')