import numpy as np

def pairwise_distance_matrix(points):
    num_points = points.shape[0]
    dist_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i, num_points):
            dist = np.linalg.norm(points[i] - points[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    max_distance = np.max(dist_matrix)
    return dist_matrix / max_distance if max_distance > 0 else dist_matrix


def compute_distance_matrices(cluster_avg_poses):
    distance_matrices = {}
    for label, avg_pose in cluster_avg_poses.items():
        dist_matrix = pairwise_distance_matrix(np.array(avg_pose))
        distance_matrices[label] = dist_matrix
    return distance_matrices

def prepare_distance_matrices_for_json(distance_matrices):
    distance_matrices_serializable = {}
    for label, dist_matrix in distance_matrices.items():
        dist_matrix_list = dist_matrix.tolist()
        distance_matrices_serializable[str(label)] = dist_matrix_list  # Convert label to string
    return distance_matrices_serializable



import json

def save_distance_matrices_to_json(distance_matrices_serializable, output_file_name):
    with open(output_file_name, 'w') as json_file:
        json.dump(distance_matrices_serializable, json_file, indent=1)
    print(f"Distance matrices saved to {output_file_name}")


def prepare_new_pdm(selected_avg_clusters, handshape, output_file_name):
    from pdm import compute_distance_matrices, prepare_distance_matrices_for_json, save_distance_matrices_to_json
    # Assuming 'cluster_avg_poses' is your dictionary of average poses

    # Step 2: Compute the distance matrices
    #from load_data import load_json
    #selected_avg_clusters = load_json('references/1_ref_pos.json')

    distance_matrices = compute_distance_matrices(selected_avg_clusters)

    # Step 3: Prepare data for JSON serialization
    distance_matrices_serializable = prepare_distance_matrices_for_json(distance_matrices)

    # Step 4: Save the distance matrices to a JSON file

    
    save_distance_matrices_to_json(distance_matrices_serializable, output_file_name)
