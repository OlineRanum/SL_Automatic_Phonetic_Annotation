import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.dataloader import DataLoader

# Directory containing the CSV files
base = '/home/oline/3D_MoCap/'
input_directory = base + 'data/'
output_directory = base +'figs/motion_plots/'
summary_file = base + 'activity_summary.csv'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Initialize list to store activity indices for each file
activity_summary = []

# Process each CSV file in the directory
for file in os.listdir(input_directory):
    if file.endswith('.csv'):  # Process only CSV files
        file_path = os.path.join(input_directory, file)
        print(f"Processing file: {file_path}")

        # Load the data
        loader = DataLoader(file_path, mode='fullpose')
        df = loader.load_data()

        # Extract the wrist location data
        wrist_location = loader.get_keypoint(df, 'ROWR', mask_nans=True)

        # Calculate the norm across the vertical axis, using masked arrays
        norms = np.sqrt(np.nansum(np.square(wrist_location), axis=1))

        # Min-max normalization of norms, respecting masked values
        min_val = norms.min()  # Minimum value in the norms (ignoring masks)
        max_val = norms.max()  # Maximum value in the norms (ignoring masks)
        normalized_norms = (norms - min_val) / (max_val - min_val)

        # Define threshold for activity detection (in normalized scale)
        threshold = 0.95  # Adjust based on normalized data

        # Determine active and resting frames
        is_active = normalized_norms > threshold  # Boolean array: True if active, False if resting

        # Get the first and last indices where is_active is True
        active_indices = np.where(is_active)[0]

        if active_indices.size > 0:
            first_index = active_indices[0]
            last_index = active_indices[-1]

            # Adjust the activity index
            first_index = min(first_index + 100, len(normalized_norms) - 1)  # Add 100 steps later
            last_index = max(last_index - 100, 0)  # Subtract 100 steps earlier

            # Update the is_active array
            is_active[:] = False  # Reset to all False
            is_active[first_index:last_index + 1] = True  # Set adjusted active range
        else:
            first_index = -1
            last_index = -1
            print(f"No active frames found in {file}")

        # Store the results in the summary list
        activity_summary.append({'filename': file, 'first_index': first_index, 'last_index': last_index})

        # Plot the results
        
        plt.figure(figsize=(10, 6))

        # Color coding based on activity
        for i in range(len(normalized_norms) - 1):  # Loop through frames
            color = 'blue' if is_active[i] else 'orange'  # Set color based on activity
            plt.plot([i, i + 1],
                     [normalized_norms[i], normalized_norms[i + 1]],
                     color=color)

        plt.title(f'Normalized Wrist Norm for {file} (Blue: Active, Orange: Resting)')
        plt.xlabel('Frame')
        plt.ylabel('Normalized Norm')
        plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
        plt.legend()

        # Save the plot
        plot_filename = os.path.join(output_directory, f"{os.path.splitext(file)[0]}_plot.png")
        plt.savefig(plot_filename)
        plt.close()  # Close the figure to free memory

# Save the summary to a CSV file
summary_df = pd.DataFrame(activity_summary)
summary_df.to_csv(summary_file, index=False)
print(f"Activity summary saved to {summary_file}")
