import numpy as np 

def pairwise_distance_matrix(points):
    """
    Calculate the normalized symmetric pairwise distance matrix for a given 3D point cloud.
    
    Parameters:
        points (numpy.ndarray): An array of shape (21, 3) representing the 3D coordinates of the hand nodes.
        
    Returns:
        numpy.ndarray: A normalized symmetric pairwise distance matrix of shape (21, 21).
    """

    # Number of nodes
    num_points = points.shape[0]
    
    # Initialize the distance matrix
    dist_matrix = np.zeros((num_points, num_points))
    
    # Calculate pairwise Euclidean distances manually
    for i in range(num_points):
        for j in range(i, num_points):  # Calculate only upper triangle
            dist = np.linalg.norm(points[i] - points[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # Ensure symmetry
    
    # Normalize by the maximum distance
    max_distance = np.max(dist_matrix)
    if max_distance > 0:
        normalized_dist_matrix = dist_matrix / max_distance
    else:
        normalized_dist_matrix = dist_matrix  # If max distance is 0, avoid division

    return normalized_dist_matrix

# List of keys to keep

def calculate_quaternions(keypoints):
    """
    Given a (21, 3) matrix of keypoints in 3D, calculate the quaternion representing
    the rotation for each segment between consecutive keypoints.
    
    Parameters:
    - keypoints: np.ndarray of shape (21, 3), representing 3D keypoints.
    
    Returns:
    - quaternions: List of quaternions (as [w, x, y, z] arrays) for each segment.
    """
    # List to store quaternions for each segment
    quaternions = []
    
    # Define a reference vector (e.g., pointing along x-axis)
    reference_vector = np.array([1, 0, 0])

    # Loop through each consecutive pair of keypoints
    for i in range(len(keypoints) - 1):
        # Calculate the direction vector for the current segment
        segment_vector = keypoints[i + 1] - keypoints[i]
        segment_vector /= np.linalg.norm(segment_vector)  # Normalize it

        # Calculate rotation quaternion from reference vector to segment vector

        rotation = R.from_rotvec(np.cross(reference_vector, segment_vector))
        quaternion = rotation.as_quat()  # Format: [x, y, z, w]
        
        # Reorder to [w, x, y, z] and append to list
        quaternions.append([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])

    return np.array(quaternions)