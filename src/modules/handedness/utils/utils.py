import numpy as np 


def calculate_center_of_mass(pose):
    # Calculate the center of mass for each frame
    return np.ma.mean(pose, axis=1)  # Average of the 21 keypoints in each frame

def calculate_velocity(center_of_mass):
    # Calculate the velocity as the difference between consecutive centers of mass
    velocity = np.diff(center_of_mass, axis=0)
    return velocity

def integrate(arr):
    # Calculate the magnitude of the array
    
    arr_mag = np.sqrt(np.sum(arr**2))  # Use np.sqrt and sum for norm
    
    # Integrate the velocity over time (sum of magnitudes)
    integrated_arr = np.sum(arr_mag)
    return sum(integrated_arr)

def get_masked_arr(pose, conf):
    conf = np.reshape(conf, (conf.shape[0], conf.shape[2], conf.shape[1]))
    pose = np.ma.masked_equal(pose * conf, 0)  # Mask positions where values are 0
    return pose, conf

import numpy as np

def get_nan_arr(pose, conf):
    """
    Replace positions in the pose array where the confidence-adjusted value is zero with np.nan.
    
    Parameters:
    - pose (np.ndarray): Array of pose keypoints, shape [T, K, 3]
    - conf (np.ndarray): Confidence scores, needs to be reshaped to align with pose
    
    Returns:
    - pose_nan (np.ndarray): Pose array with zeros replaced by np.nan
    - conf (np.ndarray): Reshaped confidence scores
    """
    # Reshape confidence scores to match pose dimensions
    conf = np.reshape(conf, (conf.shape[0], conf.shape[2], conf.shape[1]))
    
    # Multiply pose by confidence to zero out low-confidence keypoints
    pose_conf = pose * conf
    
    # Replace zero values with np.nan
    pose_nan = np.where(pose_conf == 0, np.nan, pose_conf)
    
    return pose_nan, conf


def get_normalized_coord(com, axis = 1):
    """ Recenter the coordinates to zero and flip the y-axis
    """
#    return -com[:,axis] - np.min(-com[:,axis])
    return abs(com[:,axis] - com[0,axis])


def extract_names_from_filtered_file(filtered_file_path):
    filtered_names = set()
    
    # Read the filtered file and extract names after the last underscore in the second column
    with open(filtered_file_path, 'r') as f:
        for line in f:
            columns = line.strip().split(',')
            if len(columns) >= 2:
                name_with_underscores = columns[1].strip()
                name = name_with_underscores.split('_')[-1]
                filtered_names.add(name)
    
    return filtered_names


