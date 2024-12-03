import numpy as np
from tqdm import tqdm
import os
from PoseTools.data.parsers_and_processors.parsers import PoseFormatParser, TxtParser
from PoseTools.data.parsers_and_processors.processors import MediaPipeProcessor
import pandas as pd
from PoseTools.src.utils.preprocessing import PoseSelect


import numpy as np

def normalize_pose(pose, reference_keypoint=0, scaling_keypoints=(11, 12)):
    """
    Normalize the pose data by centering and scaling.

    Parameters:
    - pose: NumPy array of shape [T, K, 3]
    - reference_keypoint: Integer index of the keypoint to center the pose
    - scaling_keypoints: Tuple of two keypoint indices to compute scaling factor

    Returns:
    - normalized_pose: NumPy array of shape [T, K, 3]
    """
    normalized_pose = pose.copy()

    # Extract x and y coordinates
    x = normalized_pose[:, :, 0]
    y = normalized_pose[:, :, 1]

    # **Translation Normalization (Centering)**
    # Get the coordinates of the reference keypoint
    ref_x = x[:, reference_keypoint].reshape(-1, 1)  # Shape: [T, 1]
    ref_y = y[:, reference_keypoint].reshape(-1, 1)  # Shape: [T, 1]

    # Subtract the reference keypoint coordinates from all keypoints
    normalized_pose[:, :, 0] = x - ref_x
    normalized_pose[:, :, 1] = y - ref_y

    # **Scale Normalization**
    # Calculate the distance between the two scaling keypoints for each frame
    kp1 = normalized_pose[:, scaling_keypoints[0], :2]  # Shape: [T, 2]
    kp2 = normalized_pose[:, scaling_keypoints[1], :2]  # Shape: [T, 2]
    distances = np.linalg.norm(kp1 - kp2, axis=1)  # Shape: [T]

    # To avoid division by zero, set minimum distance
    distances[distances == 0] = 1e-6

    # Compute scaling factors (inverse of distances)
    scaling_factors = 1 / distances
    scaling_factors = scaling_factors.reshape(-1, 1, 1)  # Shape: [T, 1, 1]

    # Apply scaling to x and y coordinates
    normalized_pose[:, :, 0:2] *= scaling_factors

    return normalized_pose

def process_pose_file(pose_path):
    """
    Load and preprocess the pose data from a given file path.
    """
    pose_loader = PoseFormatParser(pose_path)
    pose, conf = pose_loader.read_pose()
    
    if pose is not None:
        pose_selector = PoseSelect(preset="mediapipe_holistic_minimal_27")
        pose = pose_selector.clean_keypoints(pose)
        pose = pose_selector.get_keypoints_pose(pose)
        pose = normalize_pose(pose, reference_keypoint=0, scaling_keypoints=(11, 12))
        return pose
    else:
        return None

def get_wrist_stats(pose):
    """
    Calculate the mean and standard deviation of the wrist locations.
    
    Parameters:
    - pose: NumPy array of shape [T, K, 3]
    
    Returns:
    - Dictionary containing mean and std dev for left and right wrists.
    """
    stats = {}
    
    wrists = {
        'right': 16,  # Index for the right wrist
        'left': 15    # Index for the left wrist
    }
    
    for wrist, idx in wrists.items():
        x = pose[:, idx, 0]
        y = pose[:, idx, 1]
        
        stats[f'{wrist}_x_mean'] = np.mean(x)
        stats[f'{wrist}_x_std'] = np.std(x)
        stats[f'{wrist}_y_mean'] = np.mean(y)
        stats[f'{wrist}_y_std'] = np.std(y)
    
    return stats

def segment_pose(pose):
    """
    Segment the pose to keep only the middle fifth of the temporal dimension.
    
    Parameters:
    - pose: NumPy array of shape [T, K, 3]
    
    Returns:
    - Sliced NumPy array of shape [T', K, 3]
    """
    T = pose.shape[0]
    start = T // 5
    end = 3 * T // 5

    # Slice the array to keep only the middle fifth of the T dimension
    return pose[start:end]

def main(df, pose_directory, max_files=None, return_plot = None):
    """
    Evaluate the average and standard deviation of wrist locations in the given DataFrame.
    
    Parameters:
    - df: Pandas DataFrame containing pose annotations.
    - pose_directory: Directory where pose files are stored.
    - max_files: Optional integer to limit the number of files processed.
    
    Returns:
    - Pandas DataFrame containing the statistics for each pose file.
    """
    filenames = [f"{file}.pose" for file in df['Annotation ID Gloss: Dutch'].values]
    
    results = []
    processed_count = 0
    
    for file in tqdm(filenames, desc="Processing pose files"):
        pose_path = os.path.join(pose_directory, file)
        pose = process_pose_file(pose_path)
        
        if pose is not None:
            # Segment the pose
            pose = segment_pose(pose)
            
            # Flip the pose upside down by inverting the y-axis
            pose[:, :, 1] *= -1
            
            # Calculate statistics for wrist locations
            stats = get_wrist_stats(pose)

            if return_plot is not None and file[:-5] == return_plot:
                return stats, pose
            
            # Optionally, add the filename or other identifiers to the stats
            stats['Annotation ID Gloss: Dutch'] = file[:-5]
            
            results.append(stats)
            processed_count += 1
            
            #print(f"Processed: {file} | Total Processed: {processed_count}")
            
            if max_files is not None and processed_count >= max_files:
                print(f"Reached the maximum limit of {max_files} files.")
                break
        else:
            print(f"Skipping file (no pose data): {file}")
            continue
    
    # Convert the results to a Pandas DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df, None 

if __name__ == "__main__":
    # Example usage:
    # Assume you have a DataFrame 'df' with a column 'Annotation ID Gloss: Dutch'
    # and a directory 'pose_directory' containing the pose files.
    
    # Load your DataFrame (replace with your actual data loading method)
    # For example:
    # df = pd.read_csv('your_annotations.csv')
    
    # Define the pose directory
    # pose_directory = '/path/to/pose/files'
    
    # For demonstration, let's create a mock DataFrame
    mock_data = {
        'Annotation ID Gloss: Dutch': ['pose1', 'pose2', 'pose3']  # Replace with actual IDs
    }
    df = pd.DataFrame(mock_data)
    
    # Define the pose directory (replace with your actual directory)
    pose_directory = '/path/to/pose/files'
    
    # Call the main function
    stats_df = main(df, pose_directory, max_files=300)  # Set max_files as needed
    
    # Display the resulting DataFrame
    print(stats_df)
    
    # Optionally, save the results to a CSV file
    # stats_df.to_csv('wrist_stats.csv', index=False)
