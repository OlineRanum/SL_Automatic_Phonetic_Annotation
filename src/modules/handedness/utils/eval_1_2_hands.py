import numpy as np
from tqdm import tqdm
import os
from PoseTools.data.parsers_and_processors.parsers import PoseFormatParser, TxtParser
from PoseTools.data.parsers_and_processors.processors import MediaPipeProcessor
from PoseTools.src.modules.handedness.utils.utils import calculate_center_of_mass, calculate_velocity, get_masked_arr, get_nan_arr, get_normalized_coord, extract_names_from_filtered_file
from PoseTools.src.modules.handedness.utils.graphics import plot_position, plot_velocity, plot_integrated_velocities, plot_position_and_velocity,plot_position_velocity_product
import pandas as pd
import os
import numpy.ma as ma  # Import masked array module



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


import numpy as np


def get_lr(pose_path):
    """
    Extract and normalize the Y coordinates for both hands from a pose file.
    Utilizes masked arrays to handle missing or invalid data.
    
    Parameters:
    - pose_path (str): Path to the pose file.
    
    Returns:
    - com_r_y (np.ma.MaskedArray): Normalized Y coordinates for the right hand.
    - com_l_y (np.ma.MaskedArray): Normalized Y coordinates for the left hand.
    """
    pose_loader = PoseFormatParser(pose_path)
    pose, conf = pose_loader.read_pose()
        
    # Get the right and left hand poses
    mp_processor = MediaPipeProcessor(pose, conf)
    pose_r, pose_l, conf_r, conf_l = mp_processor.get_hands()

    # Get masked arrays instead of NaNs
    pose_r_masked, conf_r = get_masked_arr(pose_r, conf_r)
    pose_l_masked, conf_l = get_masked_arr(pose_l, conf_l)

    # Calculate center of mass for both hands
    com_r = calculate_center_of_mass(pose_r_masked)  # Should return masked array
    com_r_y = get_normalized_coord(com_r)

    com_l = calculate_center_of_mass(pose_l_masked)  # Should return masked array
    com_l_y = get_normalized_coord(com_l)

    # Normalize positions jointly
    com_r_y, com_l_y = minmax_normalize_together(com_r_y, com_l_y)

    # Align the lengths of com_r_y and com_l_y by masking the extra elements
    len_r = len(com_r_y)
    len_l = len(com_l_y)

    if len_r > len_l:
        # Mask extra elements in com_l_y
        padding = ma.masked_all(len_r - len_l)
        com_l_y = ma.concatenate([com_l_y, padding])
    elif len_l > len_r:
        # Mask extra elements in com_r_y
        padding = ma.masked_all(len_l - len_r)
        com_r_y = ma.concatenate([com_r_y, padding])

    return com_r_y, com_l_y

def detect_active_hands_sliding_window(com_r_y, com_l_y, window_size=2, threshold=0.3):
    """
    Detect active hands in pose data using a sliding window approach.
    
    Parameters:
    - com_r_y (np.ndarray): Array of right hand y positions over time.
    - com_l_y (np.ndarray): Array of left hand y positions over time.
    - window_size (int): Number of frames in each sliding window.
    - threshold (float): Relative distance threshold to resting position to consider as active.
    
    Returns:
    - active_r (np.ndarray): Boolean array indicating active frames for the right hand.
    - active_l (np.ndarray): Boolean array indicating active frames for the left hand.
    """
    # Ensure inputs are NumPy arrays
    com_r_y = np.array(com_r_y)
    com_l_y = np.array(com_l_y)
    
    # Validate input lengths
    if len(com_r_y) != len(com_l_y):
        raise ValueError("Right and left hand coordinate arrays must be of the same length.")
    
    # Handle np.nan by replacing them with the resting position
    initial_window_r = com_r_y[:window_size]
    initial_window_l = com_l_y[:window_size]
    
    # Compute resting positions using nanmean to ignore np.nan
    resting_r = np.nanmean(initial_window_r)
    resting_l = np.nanmean(initial_window_l)
    
    # Replace np.nan with resting positions
    com_r_y = np.where(np.isnan(com_r_y), resting_r, com_r_y)
    com_l_y = np.where(np.isnan(com_l_y), resting_l, com_l_y)
    
    # Compute absolute deviations from the resting position
    dev_r = np.abs(com_r_y - resting_r)
    dev_l = np.abs(com_l_y - resting_l)
    
    # Calculate number of windows
    num_windows = len(com_r_y) - window_size + 1
    
    if num_windows <= 0:
        raise ValueError("Pose data is shorter than the window size.")
    
    # Create sliding windows using stride tricks for efficiency
    try:
        # Available in NumPy >= 1.20
        windows_dev_r = np.lib.stride_tricks.sliding_window_view(dev_r, window_shape=window_size)
        windows_dev_l = np.lib.stride_tricks.sliding_window_view(dev_l, window_shape=window_size)
    except AttributeError:
        # For older NumPy versions, use a manual sliding window approach
        def sliding_window(arr, window_size):
            return np.array([arr[i:i+window_size] for i in range(len(arr) - window_size +1)])
        
        windows_dev_r = sliding_window(dev_r, window_size)
        windows_dev_l = sliding_window(dev_l, window_size)
    
    # Calculate the maximum deviation within each window
    max_dev_r = np.nanmax(windows_dev_r, axis=1)
    max_dev_l = np.nanmax(windows_dev_l, axis=1)
    
    # Determine active windows based on the threshold
    active_windows_r = max_dev_r > threshold
    active_windows_l = max_dev_l > threshold
    
    # Initialize boolean arrays for active frames
    active_r = np.zeros_like(dev_r, dtype=bool)
    active_l = np.zeros_like(dev_l, dtype=bool)
    
    # Map window-based activity to frame-based activity
    for i in range(num_windows):
        if active_windows_r[i]:
            # Ensure the slice does not exceed the array bounds
            end_index_r = min(i + window_size, len(active_r))
            active_r[i:end_index_r] = True
        if active_windows_l[i]:
            # Ensure the slice does not exceed the array bounds
            end_index_l = min(i + window_size, len(active_l))
            active_l[i:end_index_l] = True
    
    return active_r, active_l

def calculate_velocity(position_array):
    """
    Calculate the velocity based on the position array.
    Velocity is approximated as the difference between consecutive positions.
    
    Parameters:
    - position_array (np.ndarray): Array of positions over time.
    
    Returns:
    - velocity (np.ndarray): Array of velocities.
    """
    velocity = np.diff(position_array, prepend=position_array[0])
    return velocity

def process_pose_file(pose_path, process_single_file=False, mode='word', window_size=2, threshold=0.1, activation_arrays = None):
    """
    Load, preprocess, normalize, and analyze the pose data from a given file path.
    
    Parameters:
    - pose_path (str): Path to the pose file.
    - process_single_file (bool): If True, generates plots for the pose.
    - mode (str): 'word' for word-level analysis, 'sentence' for sentence-level analysis, 'wild' for future implementation.
    - window_size (int): Window size for sliding window (used in 'sentence' mode).
    - threshold (float): Threshold for activity detection (used in 'sentence' mode).
    
    Returns:
    - filename (str): Name of the processed pose file.
    - integrated_r (float or int): Integrated velocity (word) or active frame count (sentence) for the right hand.
    - integrated_l (float or int): Integrated velocity (word) or active frame count (sentence) for the left hand.
    """
    # Extract filename without extension
    filename = os.path.basename(pose_path).replace('.pose', '')
    
    # Get normalized y-coordinates for both hands
    com_r_y, com_l_y = get_lr(pose_path)
    print(np.sum(com_r_y), np.sum(com_l_y))
    
    
    if com_r_y.size == 0 or com_l_y.size == 0:
        print(f"No valid hand data for {filename}. Skipping.")
        return filename, 0, 0
    
    # Initialize velocities
    vel_r_mag_norm = []
    vel_l_mag_norm = []
    pos_vel_r = []
    pos_vel_l = []
    
    if mode == 'word':
        if process_single_file:
            # Plot the y-coordinates positions without activity flags
            # Since 'word' mode doesn't involve activity detection, pass all frames as active
            plot_position_velocity_product(
                pos_r=com_r_y, 
                pos_l=com_l_y, 
                active_r=[True]*len(com_r_y), 
                active_l=[True]*len(com_l_y), 
                vel_r=[0]*len(com_r_y),  # No velocity data for 'word' mode
                vel_l=[0]*len(com_l_y), 
                pos_vel_r=[0]*len(com_r_y),  # Product is zero
                pos_vel_l=[0]*len(com_l_y),
                pose_filename=filename
            )
        
        # Calculate the integrated velocity for both hands
        integrated_r = np.nansum(com_r_y)
        integrated_l = np.nansum(com_l_y)
    
    elif mode == 'sentence':
        # Detect active hands using sliding window
        active_r, active_l = detect_active_hands_sliding_window(com_r_y, com_l_y, window_size, threshold)
        print(np.sum(active_l), np.sum(active_r))
        # Calculate velocities
        velocity_r = calculate_velocity(com_r_y)
        velocity_l = calculate_velocity(com_l_y)
        
        # Calculate velocity magnitudes (Euclidean norm if multi-dimensional)
        # Since com_r_y and com_l_y are 1D, velocities are also 1D
        vel_r_mag = np.abs(velocity_r)  # Absolute velocity
        vel_l_mag = np.abs(velocity_l)
        
        # Normalize velocities jointly
        vel_r_mag_norm, vel_l_mag_norm = minmax_normalize_together(vel_r_mag, vel_l_mag)
        
        # Calculate the product of position and velocity
        pos_vel_r = com_r_y * vel_r_mag_norm
        pos_vel_l = com_l_y * vel_l_mag_norm
        
        # Summarize activity (e.g., total active frames)
        integrated_r = np.sum(active_r)
        integrated_l = np.sum(active_l)
        
        if process_single_file:
            # Plot the positions, velocities, and their products with activity flags
            plot_position_velocity_product(
                pos_r=com_r_y, 
                pos_l=com_l_y, 
                active_r=active_r, 
                active_l=active_l, 
                vel_r=vel_r_mag_norm, 
                vel_l=vel_l_mag_norm, 
                pos_vel_r=pos_vel_r, 
                pos_vel_l=pos_vel_l, 
                pose_filename=filename
            )
    
    elif mode == 'wild':
        raise NotImplementedError("Wild mode is not implemented yet.")
    
    else:
        raise ValueError('Invalid mode: Select either "word", "sentence", or "wild".')

    return filename, integrated_r, integrated_l


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

def main_handedness(boolean):
    bool_L, bool_R = boolean
    bool_L = np.array(bool_L)
    bool_R = np.array(bool_R)
    
    # Create an empty array with a larger string dtype to handle concatenated strings
    handedness = np.full_like(bool_L, '', dtype='<U12')
    
    # Compute handedness labels
    prod = bool_L * bool_R
    handedness[prod == 1] = '2 handed'
    handedness[(prod == 0) & (bool_L == 1)] = '1 handed Left'
    handedness[(prod == 0) & (bool_R == 1)] = '1 handed Right'
    handedness[(prod == 0) & (bool_L == 0) & (bool_R == 0)] = 'Resting'

    return handedness


def main_handedness_old(mode, pose_directory, pose_file = None, gloss_df=None, percent_threshold=30, motion_threshold = 0.05, test=False, activation_arrays = None):
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
    if pose_file is not None:
        pose_files = [pose_file]
    else:
        pose_files = [
            filename for filename in os.listdir(pose_directory)
            if filename.endswith(".pose") and os.path.isfile(os.path.join(pose_directory, filename))
        ]

    # Use tqdm to create a progress bar for iterating through the filtered list of .pose files
    for filename in tqdm(pose_files, desc="Processing Pose Files"):
        if activation_arrays is not None:
            if np.sum(activation_arrays[0]) == 0:
                L_hand = 'Inactive'
            else:
                L_hand = 'Active'
            if np.sum(activation_arrays[1]) == 0:
                R_hand = 'Inactive'
            else:
                R_hand = 'Active'
            if L_hand == 'Inactive' and R_hand == 'Inactive':
                status = 'None'
            elif L_hand == 'Active' and R_hand == 'Active':
                status = 'Both'
            elif L_hand == 'Active' and R_hand == 'Inactive':
                status = 'Left'
            elif L_hand == 'Inactive' and R_hand == 'Active':
                status = 'Right'
            if np.sum(activation_arrays[0]) > np.sum(activation_arrays[1]):
                percent_difference = np.sum(activation_arrays[1]) / np.sum(activation_arrays[0]) * 100
            elif np.sum(activation_arrays[1]) > np.sum(activation_arrays[0]):
                percent_difference = np.sum(activation_arrays[0]) / np.sum(activation_arrays[1]) * 100
            return {filename: [status, percent_difference]} 
        else: 
            file_path = os.path.join(pose_directory, filename)
            
            # Extract the gloss name (assuming filename format "gloss_name.pose")
            sign_name = os.path.splitext(filename)[0]
            
            # Skip if test=True and sign_name is not in gloss_set
            if test and gloss_set is not None and sign_name not in gloss_set:
                continue
            
            if mode == 'word':
                # Process the pose file to get sign_name, integrated_r, and integrated_l
                sign_name, integrated_r, integrated_l = process_pose_file(file_path, True,mode = mode)
                
                # Determine handedness
                active_hand, percent_difference = determine_handedness(sign_name, integrated_r, integrated_l, percent_threshold)
                handedness_dict[sign_name] = [active_hand, percent_difference]
            elif mode == 'sentence':
                # Process the pose file to get sign_name, integrated_r, and integrated_l
                sign_name, integrated_r, integrated_l = process_pose_file(file_path, True, mode = mode, threshold=motion_threshold)
                
                active_hand, percent_difference = determine_handedness(sign_name, integrated_r, integrated_l, percent_threshold)
                
                handedness_dict[filename] = [active_hand, percent_difference]

            else:
                print('Invalid mode: Select either "isolated" or "sentence".')
                exit()
        
        return handedness_dict

def main_handedness_inference(mode, pose_file, gloss_df=None, percent_threshold=30, test=False):
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
    pose_files = [pose_file]

    # Use tqdm to create a progress bar for iterating through the filtered list of .pose files
    for filename in tqdm(pose_files, desc="Processing Pose Files"):
        file_path = os.path.join(pose_directory, filename)
        
        # Extract the gloss name (assuming filename format "gloss_name.pose")
        sign_name = os.path.splitext(filename)[0]
        
        # Skip if test=True and sign_name is not in gloss_set
        if test and gloss_set is not None and sign_name not in gloss_set:
            continue

        if mode == 'word':
            # Process the pose file to get sign_name, integrated_r, and integrated_l
            sign_name, integrated_r, integrated_l = process_pose_file(file_path, True,mode = mode)
            
            # Determine handedness
            active_hand, percent_difference = determine_handedness(sign_name, integrated_r, integrated_l, percent_threshold)
            handedness_dict[sign_name] = [active_hand, percent_difference]
        elif mode == 'sentence':
            # Process the pose file to get sign_name, integrated_r, and integrated_l
            sign_name, integrated_r, integrated_l = process_pose_file(file_path, True, mode = mode)
            

        else:
            print('Invalid mode: Select either "isolated" or "sentence".')
            exit()
    
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

