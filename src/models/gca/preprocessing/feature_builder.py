import numpy as np
import pickle
import os
import time
import torch 
def calculate_distances(array):
    N_keypoints, N_dims = array.shape
    distance_array = np.zeros((N_keypoints, N_keypoints))

    
    for i in range(N_keypoints):
        for j in range(N_keypoints):
            if i != j:
                # Calculate the Euclidean distance between points i and j for the current frame
                distance = np.linalg.norm(array[i] - array[j])
                distance_array[i, j] = distance

    return distance_array  # Return only upper triangular part as a vector

def compute_joint_angles(pose):
    """
    Compute angles between consecutive keypoints to capture joint orientations.
    pose: Tensor of shape [n_keypoints, 3]
    """
    
    n_keypoints = pose.shape[0]
    angles = []
    for i in range(1, n_keypoints - 1):
        vec1 = pose[i] - pose[i - 1]  # Vector from previous point
        vec2 = pose[i + 1] - pose[i]  # Vector to next point
        cos_theta = torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))
        angle = torch.acos(cos_theta)  # Compute angle in radians
        angles.append(angle.item())
    return torch.tensor(angles)

def calculate_direction_vectors(array):
    N_frames, N_keypoints, N_dims = array.shape
    direction_array = np.zeros((N_frames, N_keypoints, N_keypoints, N_dims))

    for frame in range(N_frames):
        for i in range(N_keypoints):
            for j in range(N_keypoints):
                if i != j:
                    # Calculate the normalized direction vector from keypoint i to keypoint j
                    direction = array[frame, j] - array[frame, i]
                    direction_array[frame, i, j] = direction / np.linalg.norm(direction)

    return direction_array

def calculate_angles(array):
    N_frames, N_keypoints, N_dims = array.shape
    angles_array = np.zeros((N_frames, N_keypoints, N_keypoints, N_keypoints))

    for frame in range(N_frames):
        for i in range(N_keypoints):
            for j in range(N_keypoints):
                for k in range(N_keypoints):
                    if i != j and i != k:
                        # Calculate the cosine of the angle between vectors ij and ik
                        vector_ij = array[frame, j] - array[frame, i]
                        vector_ik = array[frame, k] - array[frame, i]
                        cosine_angle = np.dot(vector_ij, vector_ik) / (np.linalg.norm(vector_ij) * np.linalg.norm(vector_ik))
                        angles_array[frame, i, j, k] = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return angles_array

def calculate_cross_products(array):
    if array.shape[2] != 3:  # Cross products only apply to 3D data
        raise ValueError("Cross products are only valid for 3D coordinates.")
    
    N_frames, N_keypoints, N_dims = array.shape
    cross_product_array = np.zeros((N_frames, N_keypoints, N_keypoints, N_dims))

    for frame in range(N_frames):
        for i in range(N_keypoints):
            for j in range(N_keypoints):
                if i != j:
                    for k in range(N_keypoints):
                        if i != k and j != k:
                            # Calculate the cross product of vectors ij and ik
                            vector_ij = array[frame, j] - array[frame, i]
                            vector_ik = array[frame, k] - array[frame, i]
                            cross_product = np.cross(vector_ij, vector_ik)
                            cross_product_array[frame, i, j] = cross_product

    return cross_product_array

def combine_features(distances  = None, directions = None, angles = None, cross_products = None):
    N_frames, N_keypoints, _, N_dims = directions.shape
    
    # Flatten the features along the N_keypoints axis and combine into a single array
    if distances is not None:
        distances_flat = distances.reshape(N_frames, N_keypoints, -1)
    else:
        distances_flat = None
    if directions is not None:
        directions_flat = directions.reshape(N_frames, N_keypoints, -1)
    else:
        directions_flat = None
    if angles is not None:
        angles_flat = angles.reshape(N_frames, N_keypoints, -1)
    else:
        angles_flat = None
    if cross_products is not None:
        cross_products_flat = cross_products.reshape(N_frames, N_keypoints, -1)
    else:    
        cross_products_flat = None

    # Combine all features along the feature dimension
    features = []
    for feat in [distances_flat, directions_flat, angles_flat, cross_products_flat]:
        if feat is not None:
            features.append(feat)
    combined_features = np.concatenate(features, axis=2)
    
    return combined_features

def process_all_poses(input_folder, output_folder, pose_type='pose_format'):
    """
    Iterate over all pose files in the input folder, preprocess them, and save them to the output folder.
    Track the time it takes to calculate each feature component.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".pkl"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            if os.path.exists(output_path):
                #print(f"{filename} already exists in the output directory. Skipping...")
                continue

            # Load the pose data from the input file
            with open(input_path, 'rb') as file:
                data = pickle.load(file)['keypoints']

            

            try:
                    # Track time for calculating distances
                start_time = time.time()
                distances = calculate_distances(data)
                dist_time = time.time() - start_time
                #print(f"Time taken to calculate distances for {filename}: {dist_time:.4f} seconds")

                # Track time for calculating directions
                start_time = time.time()
                directions = calculate_direction_vectors(data)
                dir_time = time.time() - start_time
                #print(f"Time taken to calculate directions for {filename}: {dir_time:.4f} seconds")

                # Track time for calculating angles
                start_time = time.time()
                #angles = calculate_angles(data)
                angle_time = time.time() - start_time
                #print(f"Time taken to calculate angles for {filename}: {angle_time:.4f} seconds")

                # Track time for calculating cross products
                start_time = time.time()
                #cross_products = calculate_cross_products(data)
                cross_time = time.time() - start_time
                #print(f"Time taken to calculate cross products for {filename}: {cross_time:.4f} seconds")

                # Combine all features into one array
                combined_features = combine_features(distances, directions)
                # Save the processed combined data back to the output folder under a single 'keypoints' key
                with open(output_path, 'wb') as output_file:
                    pickle.dump({'keypoints': combined_features}, output_file)
                
                #print(f"Processed {filename} and saved to {output_path}\n")
            except Exception as e:
                distances = calculate_distances(data)
                directions = calculate_direction_vectors(data)
                distances =  np.pad(distances, ((0, 0), (0, 0), (0, 63)), mode='constant', constant_values=0)

                with open(output_path, 'wb') as output_file:
                    pickle.dump({'keypoints': distances}, output_file)
                print(filename, e)
                continue
if __name__ == "__main__":
    input_folder = "../GMVISR/data/hamer_pkl"
    output_folder = "../GMVISR/data/hamer_pkl_multi"
    process_all_poses(input_folder, output_folder, pose_type='hamer')

