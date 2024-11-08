def plot_cm(true_labels, pred_labels, save_path='/home/gomer/oline/PoseTools/src/modules/handshapes/utils/confusion_matrix.png'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import numpy as np
    
    # Generate confusion matrix with unique labels from both lists
    unique_labels = sorted(set(true_labels) | set(pred_labels))
    conf_matrix = confusion_matrix(true_labels, pred_labels, labels=unique_labels)

    # Convert each row to percentages and handle rows with a sum of zero
    conf_matrix_percentage = np.zeros_like(conf_matrix, dtype=float)
    for i, row in enumerate(conf_matrix):
        row_sum = row.sum()
        if row_sum > 0:
            conf_matrix_percentage[i] = (row / row_sum) * 100

    # Round to integers for display without decimals
    conf_matrix_percentage = conf_matrix_percentage.round(0).astype(int)

    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_percentage, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=unique_labels, yticklabels=unique_labels)

    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.title("Confusion Matrix (Percentage)")
    plt.savefig(save_path)
