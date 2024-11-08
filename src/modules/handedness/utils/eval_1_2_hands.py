import numpy as np
from tqdm import tqdm
import os
from PoseTools.data.parsers_and_processors.parsers import PoseFormatParser, TxtParser
from PoseTools.data.parsers_and_processors.processors import MediaPipeProcessor
from PoseTools.src.modules.handedness.utils.utils import calculate_center_of_mass, calculate_velocity, get_masked_arr, get_normalized_coord, extract_names_from_filtered_file
from PoseTools.src.modules.handedness.utils.graphics import plot_position, plot_velocity, plot_integrated_velocities
import pandas as pd

def minmax_normalize_together(arr1, arr2):
    # Calculate the global minimum and maximum across both arrays
    min_val = min(np.min(arr1), np.min(arr2))
    max_val = max(np.max(arr1), np.max(arr2))
    # Avoid division by zero if all values are the same
    if max_val - min_val == 0:
        return np.zeros_like(arr1), np.zeros_like(arr2)
    # Normalize both arrays with the global min and max
    norm_arr1 = (arr1 - min_val) / (max_val - min_val)
    norm_arr2 = (arr2 - min_val) / (max_val - min_val)
    return norm_arr1, norm_arr2

def process_pose_file(pose_path, process_single_file = False):
    pose_loader = PoseFormatParser(pose_path)
    pose, conf = pose_loader.read_pose()
        
    # Get the right and left hand poses

    mp_processor = MediaPipeProcessor(pose, conf)
    pose_r, pose_l, conf_r, conf_l = mp_processor.get_hands()

    # Get masked arrays
    pose_r, conf_r = get_masked_arr(pose_r, conf_r)
    pose_l, conf_l = get_masked_arr(pose_l, conf_l)

    
    # Calculate center of mass for both hands
    com_r = calculate_center_of_mass(pose_r)
    com_r_y = get_normalized_coord(com_r)

    com_l = calculate_center_of_mass(pose_l)
    com_l_y = get_normalized_coord(com_l)


    # Apply Min-Max normalization
    # Remove masked elements from com_r_y and com_l_y
    com_r_y, com_l_y = minmax_normalize_together(com_r_y, com_l_y)

    # Integrate velocity
    filename =  pose_path.split('/')[-1].replace('.pose', '')


    if process_single_file:
        # Plot the y-cppdomate
        plot_position(com_r_y.tolist(), com_l_y.tolist(), filename)
        
                # Calculate velocity profiles
        velocity_r = calculate_velocity(com_r)
        velocity_l = calculate_velocity(com_l)

        # Plot the velocity profiles    
        #plot_velocity(velocity_r, velocity_l, 'test')
    
    com_r_y = com_r_y.compressed()
    com_l_y = com_l_y.compressed()

    integrated_r = sum(com_r_y)
    integrated_l = sum(com_l_y)


    
    return filename, integrated_r, integrated_l

import os


def determine_handedness(sign, integrated_r, integrated_l, percent_threshold=50):
    # Calculate the larger and smaller of the two integrated velocities
    max_velocity = max(integrated_r, integrated_l)
    min_velocity = min(integrated_r, integrated_l)
    
    # If max_velocity is zero, return "None" or "Both" since there's no activity
    if max_velocity == 0:
        active_hand = "None"  # Alternatively, you could set this to "Both" if that makes more sense
        percent_difference = 0  # No difference if both are zero
    else:
        # Calculate the percent difference
        percent_difference = (max_velocity - min_velocity) / max_velocity * 100

        # Check if the percent difference is within the threshold
        if percent_difference <= percent_threshold:
            # If within the threshold, both hands are considered active
            active_hand = "B"
        else:
            # Otherwise, determine the more active hand
            if integrated_r > integrated_l:
                active_hand = "R"
            else:
                active_hand = "L"
    
    #print(f"{sign}: Active Hand = {active_hand}, Percent difference = {percent_difference:.2f}%\n")
    return active_hand, round(percent_difference, 2)
'''
def main_handedness(pose_directory, percent_threshold=50, test = False):
    handedness_dict = {}
    for filename in os.listdir(pose_directory):
        file_path = os.path.join(pose_directory, filename)
        # Ensure we are processing only files (skip directories)
        if os.path.isfile(file_path):
            # Process the pose file to get filename, integrated_r, and integrated_l
            sign_name, integrated_r, integrated_l = process_pose_file(file_path, True)
            # Determine and print handedness
            active_hand = determine_handedness(sign_name, integrated_r, integrated_l, percent_threshold)
            handedness_dict[sign_name] = active_hand
    
    return handedness_dict
'''
import os

def main_handedness(pose_directory, gloss_df=None, percent_threshold=30, test=False):
    """
    Determine the handedness for pose files in a directory.
    
    Parameters:
    - pose_directory (str): Path to the directory containing pose files.
    - gloss_df (DataFrame): DataFrame containing glosses to evaluate if test=True.
    - percent_threshold (int): Threshold percentage to determine handedness.
    - test (bool): If True, only evaluates files listed in gloss_df. If False, evaluates all files in pose_directory.
    
    Returns:
    - handedness_dict (dict): Dictionary with sign names as keys and determined handedness as values.
    """
    from tqdm import tqdm
    handedness_dict = {}
    
    # Create a set of gloss names from gloss_df if test=True
    if test and gloss_df is not None:
        gloss_set = set(gloss_df['Annotation ID Gloss: Dutch'])
    else:
        gloss_set = None

    # List all valid .pose files in the directory for faster iteration
    pose_files = [
        filename for filename in os.listdir(pose_directory)
        if filename.endswith(".pose") and os.path.isfile(os.path.join(pose_directory, filename))
    ]

    # Use tqdm to create a progress bar for iterating through the filtered list of .pose files
    for filename in tqdm(pose_files, desc="Processing Pose Files"):
        file_path = os.path.join(pose_directory, filename)
        
        # Extract the gloss name (assuming filename format "gloss_name.pose")
        sign_name = os.path.splitext(filename)[0]
        
        # Skip if test=True and sign_name is not in gloss_set
        if test and gloss_set is not None and sign_name not in gloss_set:
            continue

        # Process the pose file to get sign_name, integrated_r, and integrated_l
        sign_name, integrated_r, integrated_l = process_pose_file(file_path, True)
        
        # Determine handedness
        active_hand, percent_difference = determine_handedness(sign_name, integrated_r, integrated_l, percent_threshold)
        handedness_dict[sign_name] = [active_hand, percent_difference]
    
    return handedness_dict

import pandas as pd

def evaluate_handedness(handedness_dict, metadata):
    """
    Evaluate the accuracy of handedness predictions in handedness_dict against metadata.
    
    Parameters:
    - handedness_dict (dict): Dictionary with sign names as keys and predicted handedness as values.
    - metadata (DataFrame): DataFrame with a `Handedness` column containing ground truth labels for each sign.
    
    Returns:
    - accuracy (float): The overall accuracy of handedness predictions.
    - mismatches (dict): Dictionary of sign names where predictions and ground truth do not match, with details.
    """
    total = 0
    correct = 0
    
    mismatches = {}

    for sign_name, predicted_handedness in handedness_dict.items():
        # Look up the ground truth handedness for the sign in the metadata DataFrame
        ground_truth_row = metadata[metadata['Annotation ID Gloss: Dutch'] == sign_name]

        # Ensure the sign exists in metadata
        if ground_truth_row.empty:
            print(f"Warning: {sign_name} not found in metadata.")
            continue

        # Get the ground truth handedness label
        ground_truth_handedness = ground_truth_row.iloc[0]['Handedness']

        # Define the expected handedness based on the ground truth
        if ground_truth_handedness == '1':
            expected_handedness = ["L", "R"]
        elif str(ground_truth_handedness).startswith('2'):
            expected_handedness = ["B"]
        else:
            if ground_truth_handedness != -1:
                print(f"Warning: Unrecognized handedness label '{ground_truth_handedness}' for {sign_name}")
            continue

        # Compare predicted handedness with expected handedness
        if predicted_handedness in expected_handedness:
            correct += 1
        else:
            # Record mismatches for detailed analysis
            mismatches[sign_name] = {
                'predicted': predicted_handedness,
                'expected': ground_truth_handedness,
                'percentage': percent_difference
            }
        
        total += 1

    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    mismatch = len(mismatches) / total if total > 0 else 0
    print(f"Total Signs Evaluated: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Correct Mismatch: {mismatch}")
    
    return accuracy, mismatches

