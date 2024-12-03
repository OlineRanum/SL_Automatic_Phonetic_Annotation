import argparse
import os
import pickle
import re
from collections import Counter

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from PoseTools.src.modules.features.feature_transformations import pairwise_distance_matrix
from PoseTools.src.modules.handshapes.utils.build_references.graphics import plot_cm
from PoseTools.src.modules.handedness.utils.graphics import read_dict_from_txt
#from PoseTools.src.modules.segmentation.segmentation import main_segmentation
from tqdm import tqdm

def read_ground_truth_labels(sign_list_file):
    """
    Reads ground truth labels from a CSV file.

    Args:
        sign_list_file (str): Path to the sign list CSV file.

    Returns:
        dict: A dictionary mapping filenames to their labels.
    """
    ground_truth = {}
    handshapes = []
    exclude_handshapes = ['-1', 'SHS', 'WHS', 'C2_spread',  'Beak2_open', 'T_open', 'F']
    with open(sign_list_file, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 5:
                filename, handshape_label, weak_hand, handedness, hs_change = parts
                if weak_hand in ['B_curved', 'B_bent']:
                    weak_hand = 'B'
                if handshape_label in ['B_curved', 'B_bent']:
                    handshape_label = 'B'
                if filename == 'BAL-A':
                    handshape_label = 'C'
                if filename in ['VOETBALLEN-B', 'BOOS-A']:
                    weak_hand = '5'
                if filename == 'EI-B':
                    handshape_label = 'Baby_C'
                    handedness = '2s'
                if filename in ['KLEREN-A', 'RUSTIG', 'LICHAAM-B', 'ALLEMAAL-A', 'WACHTEN-A', 'LOPEN-E']:
                    handshape_label = 'B'
                if filename in ['LICHAAM-A']: # 'LEKKER-A'
                    handshape_label = '5'
                if filename in ['KLEREN-B', 'BANAAN-B']:
                    handshape_label = 'Money'
                if filename in ['NU-B', 'WETEN-B', 'PROBEREN-B']:
                    handshape_label = '1'
                if filename == 'STRAKS-A':
                    handshape_label = '1_curved'
                if filename == 'NU-A':
                    handshape_label = 'N'
                if filename == 'KIJKEN-B':
                    handshape_label = 'V'
                if filename == 'AL':
                    handedness = '1'
                    handshape_label = 'B'
                if handshape_label in exclude_handshapes:
                    continue
                handshapes.append(handshape_label)
                handshapes.append(weak_hand)
                ground_truth[filename] = {
                    'handshape': handshape_label,
                    'weak': weak_hand,
                    'handedness': handedness,
                    'hs_change': hs_change
                }
    print(f"Poses loaded from test set: {set(handshapes)}")
    print(f"Number of poses loaded from test set: {len(set(handshapes))}")
    
    return ground_truth


def remove_suffix(string):
    """
    Removes trailing underscores followed by digits from a string.

    Args:
        string (str): The input string.

    Returns:
        str: The string without the trailing suffix.
    """
    pattern = r'(_\d+)$'
    return re.sub(pattern, '', string)


import numpy as np

def aggregate_initial_consecutive_duplicates(handshapes, similarities):
    """
    Aggregates the similarity scores for the initial consecutive duplicate handshapes.

    Args:
        handshapes (list of str): List of top-n handshape names.
        similarities (list of float): Corresponding similarity scores.

    Returns:
        tuple: 
            - new_handshapes (list of str): Updated list with initial duplicates aggregated.
            - new_similarities (list of float): Updated similarity scores with aggregated values.
    """
    if not handshapes or not similarities:
        return handshapes, similarities

    first_handshape = handshapes[0]
    sum_similarity = similarities[0]
    count = 1

    # Iterate over the handshapes starting from the second element
    for h, s in zip(handshapes[1:], similarities[1:]):
        if h == first_handshape:
            sum_similarity += s
            count += 1
        else:
            break  # Stop at the first non-duplicate

    if count > 1:
        # Aggregate the initial duplicates
        new_handshapes = [first_handshape] + handshapes[count:]
        new_similarities = [sum_similarity] + similarities[count:]
        return new_handshapes, new_similarities
    else:
        # No aggregation needed
        return handshapes, similarities

def calculate_euclidean_distance_with_similarity(pose, reference_poses, n=5, transformation = 'pdm'):
    """
    Calculates the Euclidean distance between a normalized and aligned pose and each reference pose,
    returning the closest and top `n` closest handshapes along with their similarity percentages.

    Args:
        pose (np.ndarray): Normalized and aligned pose array of shape (21, 3).
        reference_poses (dict): Reference poses for comparison.
            - Keys are handshape names (strings).
            - Values are lists of reference poses (each a np.ndarray of shape (21, 3)).
        n (int, optional): Number of top closest handshapes to return. Defaults to 5.

    Returns:
        tuple: 
            - closest_handshape (str): The most similar handshape.
            - top_n_handshapes (list of tuples): List containing tuples of handshape names and their similarity percentages.
    """
    distances = []
    keys = []
    
    
    # Since data is already normalized and view-aligned, proceed directly
    # TODO: FIX this
    if transformation == 'pdm':
        transformed_pose = pairwise_distance_matrix(pose)  # No transformation needed
        orientations = False
    if transformation == 'orientation':
        transformed_pose = pose
        orientations = True
    
    for key, reference_list in reference_poses.items():
        idx = -1
        for reference_pose in reference_list:
            # Calculate Euclidean distance per node, then take the mean across all nodes
            if orientations:
                distance = np.linalg.norm(transformed_pose - reference_pose.squeeze(0), axis=1).mean()
            else: 
                distance = np.linalg.norm(transformed_pose - reference_pose, axis=1).mean()
                
            distances.append(distance)
            keys.append(key)  # Assuming keys are already clean; if not, apply remove_suffix
            idx += 1
    
    # Convert lists to numpy arrays for efficient computation
    distances = np.array(distances)  # Shape: (num_references,)
    keys = np.array(keys)            # Shape: (num_references,)
    
    # Handle cases where distance might be zero to avoid division by zero
    epsilon = 1e-10
    distances_safe = distances + epsilon
    
    # Convert distances to similarities using inverse distance
    similarities = 1 / distances_safe  # Higher similarity for smaller distance
    
    # Normalize similarities to sum to 100%
    total_similarity = np.sum(similarities)
    normalized_similarities = (similarities / total_similarity) * 100
    
    
    # Get indices of the top-n largest similarities
    sorted_indices = np.argsort(normalized_similarities)[::-1][:n]
    top_n_keys = keys[sorted_indices]
    top_n_similarities = normalized_similarities[sorted_indices]
    
    
    # Identify the closest handshape (highest similarity)
    closest_handshape = top_n_keys[0]
    
    # Prepare separate lists for top-n handshapes and their similarities
    top_n_handshapes = top_n_keys.tolist()
    similarity_top_n = top_n_similarities.tolist()
    top_n_handshapes, similarity_top_n = aggregate_initial_consecutive_duplicates(top_n_handshapes, similarity_top_n)
    
    return closest_handshape, top_n_handshapes, similarity_top_n, idx

def calculate_euclidean_distance(pose, reference_poses, n=5):
    """
    Calculates the Euclidean distance between a pose and each reference pose,
    returning the closest and top `n` closest handshapes.

    Args:
        pose (np.ndarray): Pose array of shape (21, 3).
        reference_poses (dict): Reference poses for comparison.
        n (int, optional): Number of top closest handshapes to return. Defaults to 5.

    Returns:
        tuple: Closest handshape and a list of top `n` handshapes.
    """
    distances = []
    keys = []

    transformed_pose = pairwise_distance_matrix(pose)

    for key, reference_list in reference_poses.items():
        idx = -1
        for reference_pose in reference_list:
            distance = np.linalg.norm(transformed_pose - reference_pose, axis=1).mean()
            distances.append(distance)
            keys.append(remove_suffix(key))
            idx += 1

    sorted_indices = np.argsort(distances)[:n]
    top_n_handshapes = [keys[i] for i in sorted_indices]
    closest_handshape = keys[np.argmin(distances)]

    return closest_handshape, top_n_handshapes, idx


def select_hand(pose_L, pose_R, strong_pose, weak_pose=None, handedness='1', idx_L=0, idx_R=0):
    """
    Determines which hand (left or right) is stronger based on Euclidean distance.

    Args:
        pose_L (np.ndarray): Left hand pose array.
        pose_R (np.ndarray): Right hand pose array.
        strong_pose (np.ndarray): Reference strong hand pose.
        weak_pose (list, optional): List of reference weak hand poses. Defaults to None.
        handedness (str, optional): Handedness type. Defaults to '1'.
        idx_L (int, optional): Index for left hand reference pose. Defaults to 0.
        idx_R (int, optional): Index for right hand reference pose. Defaults to 0.

    Returns:
        str or tuple: Selected hand ('L' or 'R') or tuple of strengths for both hands.
    """

    transformed_L = np.array([pairwise_distance_matrix(frame) for frame in pose_L])  # Shape: (T_frames, n_keypoints, n_keypoints)
    transformed_R = np.array([pairwise_distance_matrix(frame) for frame in pose_R])  # Shape: (T_frames, n_keypoints, n_keypoints)

    # Subtract strong_pose from each frame in transformed_L and transformed_R
    difference_L = transformed_L - strong_pose[idx_L]  # Shape: (T_frames, n_keypoints, n_keypoints)
    difference_R = transformed_R - strong_pose[idx_R]  # Shape: (T_frames, n_keypoints, n_keypoints)

    # Compute the Euclidean norm over axes (1, 2) for each frame
    d_l_s_frames = np.linalg.norm(difference_L, axis=(1, 2))  # Shape: (T_frames,)
    d_r_s_frames = np.linalg.norm(difference_R, axis=(1, 2))  # Shape: (T_frames,)

    # Average the norms across all frames
    d_l_s = d_l_s_frames.mean()
    d_r_s = d_r_s_frames.mean()

    if handedness == '1':
        # Uncomment the following lines if you want dynamic selection
        # if d_l_s < d_r_s:
        #     print('Selected hand L')
        # elif d_r_s < d_l_s:
        #     print('Selected hand R')
        return 'R'  # 'L' if d_l_s < d_r_s else 'R'

    elif handedness == '2a' and weak_pose is not None:
        d_l_w = min(np.linalg.norm(transformed_L - wp, axis=(1, 2)).mean() for wp in weak_pose)
        d_r_w = min(np.linalg.norm(transformed_R - wp, axis=(1, 2)).mean() for wp in weak_pose)

        res_L = 'strong' if d_l_s < d_l_w else 'weak'
        res_R = 'strong' if d_r_s < d_r_w else 'weak'
        return {'L': res_L, 'R': res_R}

    elif handedness == '2a' and weak_pose is None:
        res_L = 'strong'
        res_R = 'strong'
        return res_L, res_R


def preprocess_pose(pose):
    """
    Converts pose data into tensors suitable for model input.

    Args:
        pose (np.ndarray): Pose data array.

    Returns:
        list: List of torch.Tensor objects for each frame.
    """
    return [torch.tensor(frame, dtype=torch.float32) for frame in pose]

import numpy as np
import matplotlib
# Use 'Agg' backend for faster, non-interactive plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting
from tqdm import tqdm
import imageio
from collections import Counter


import matplotlib.pyplot as plt
import numpy as np
import imageio
from mpl_toolkits.mplot3d import Axes3D

def plot_pose_3d(
    keypoints,
    predictions,
    output_path,
    filename,
    gloss_hand=None,
    make_fig = False
):
    """
    Plots 3D poses and creates a GIF of the handshape predictions.

    Args:
        keypoints (np.ndarray): Array of shape [T_frames, N_keypoints, 3] representing 3D coordinates.
        predictions (dict): Dictionary containing prediction data:
            - 'closest': List of top-1 predictions per frame.
            - 'top_n': List of top-n predictions per frame.
            - 'handedness': List of handedness labels per frame (if gloss_hand is provided).
            - 'strong': List of strong handshape labels per frame (if gloss_hand is provided).
            - 'weak': List of weak handshape labels per frame (if gloss_hand is provided).
        output_path (str): Path to save the output GIF.
        filename (str): Base filename for the output.
        gloss_hand (dict, optional): Ground truth gloss hand information. Defaults to None.
    """
    # Define the edges connecting keypoints for plotting
    inward_edges = [
        [1, 0], [2, 1], [3, 2], [4, 3],    # Thumb
        [5, 0], [6, 5], [7, 6], [8, 7],    # Index Finger
        [9, 0], [10, 9], [11, 10], [12, 11],  # Middle Finger
        [13, 0], [14, 13], [15, 14], [16, 15],  # Ring Finger
        [17, 0], [18, 17], [19, 18], [20, 19]   # Pinky Finger
    ]

    # Define the viewing angles for subplots
    angles = [[0, 0], [30, -30], [30, -60], [90, 90]]

    fig = plt.figure(figsize=(10, 8))
    scatters = []  # List to store scatter plot objects
    lines = []      # List to store lists of line plot objects

    for idx_angle, angle in enumerate(angles):
        ax = fig.add_subplot(2, 2, idx_angle + 1, projection='3d')

        # Initialize scatter plot with empty data
        scatter = ax.scatter([], [], [], c='b', s=10)
        scatters.append(scatter)

        # Initialize line plots
        line_plots = []
        for edge in inward_edges:
            line, = ax.plot([], [], [], 'r-', linewidth=1)
            line_plots.append(line)
        lines.append(line_plots)

        # Set fixed view and limits
        ax.view_init(elev=angle[0], azim=angle[1])
        ax.set_zlim(-0.1, 0.02)
        ax.set_xlim(-0.1, 0.15)
        ax.set_ylim(-0.1, 0.15)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Angle: {angle[0]}°, {angle[1]}°")

    # Initialize the super title as a text object for dynamic updates
    suptitle_text = fig.suptitle('', fontsize=12)

    frames = []

    # Iterate over each frame with a progress bar
    for idx, centered_frame in enumerate(tqdm(keypoints, desc="Plotting Frames", unit="frame")):
        x, y, z = centered_frame[:, 0], centered_frame[:, 1], centered_frame[:, 2]

        # Update scatter plots
        for scatter in scatters:
            scatter._offsets3d = (x, y, z)

        # Update line plots
        for line_plots in lines:
            for line, edge in zip(line_plots, inward_edges):
                start, end = edge
                line.set_data([x[start], x[end]], [y[start], y[end]])
                line.set_3d_properties([z[start], z[end]])

        # Prepare the super title with current predictions
        if gloss_hand is not None:
            # Extract ground truth for the current frame
            handedness = predictions.get('handedness', ['Unknown'])[idx]
            strong = predictions.get('strong', ['Unknown'])[idx]
            weak = predictions.get('weak', ['Unknown'])[idx]
            closest = predictions.get('closest', ['Unknown'])[idx]
            top_n = predictions.get('top_n', [['Unknown']*5])[idx]
            suptitle = (
                f'{filename} - Hnd: {handedness}, Strong: {strong}, Weak: {weak}\n'
                f'Predicted: {closest}\nTop-3: {top_n[:3]}\nTop-5: {top_n[:5]}'
            )
        else:
            closest = predictions.get('closest', ['Unknown'])[idx]
            top_n = predictions.get('top_n', [['Unknown']*5])[idx]
            suptitle = f'{filename} - Predicted: {closest}\nTop-3: {top_n[:3]}\nTop-5: {top_n[:5]}'

        suptitle_text.set_text(suptitle)

        # Draw the updated figure
        fig.canvas.draw()

        # Convert the figure to a NumPy array
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Append the image to frames
        frames.append(image)

    plt.close(fig)  # Close the figure to free memory

    if make_gif:
        if frames:
            imageio.mimsave(output_path, frames, fps=8, loop=0)
            print('Saved output at', output_path)
        else:
            print('No frames to save.')
    return frames

def predict_handshape(
    keypoints,
    reference_poses,
    output_gif_path,
    filename,
    gloss_hand=None,
    plot=True,
    boolean_arrays=None,
    transformation = 'pdm'
):
    """
    Predicts the handshape label from pose frames and optionally creates a GIF.

    Args:
        keypoints (np.ndarray): Array of shape [T_frames, N_keypoints, 3] representing 3D coordinates.
        reference_poses (dict): Reference poses for comparison.
        output_gif_path (str): Path to save the output GIF.
        filename (str): Base filename for the output.
        gloss_hand (dict, optional): Ground truth gloss hand information.
        plot (bool, optional): Whether to generate plots. Defaults to True.
        boolean_arrays (np.ndarray, optional): Array indicating specific conditions per frame. Defaults to None.

    Returns:
        tuple: 
            - If `gloss_hand` is None: (Counter for top-1 predictions, Counter for top-3 predictions, Counter for top-5 predictions, List of top-1 predictions, List of similarities)
            - If `gloss_hand` is provided: (Median top-1 prediction, Top-3 predictions list, Top-5 predictions list, Most common index, Last top_n_similarities)
    """
    # Extract ground truth details if available
    if gloss_hand is not None:
        handedness = gloss_hand.get('handedness', [])
        weak = gloss_hand.get('weak', [])
        strong = gloss_hand.get('handshape', [])

    # Initialize prediction counters and storage
    pred_top1 = []
    pred_top3 = []
    pred_top3_list = []
    pred_top5 = []
    idxs_list = []
    similarities = []

    # Prepare predictions dictionary for plotting
    predictions = {
        'closest': [],
        'top_n': []
    }

    if gloss_hand is not None:
        predictions['handedness'] = []
        predictions['weak'] = []
        predictions['strong'] = []

    # Iterate over each frame with a progress bar
    for frame_idx, frame in enumerate(tqdm(keypoints, desc="Processing Frames", unit="frame")):
        
        if np.sum(frame) < 1e-4:
            print(f"Frame {frame_idx} is empty.")
            closest = 'No data'
            top_n = 'No data'
            top_n_similarities = [0] * 5
            idx = None
        # Calculate predictions
        closest, top_n, top_n_similarities, idx = calculate_euclidean_distance_with_similarity(frame, reference_poses, n=5, transformation = transformation)

        # Apply boolean conditions if provided
        if boolean_arrays is not None:
            if frame_idx < len(boolean_arrays) and boolean_arrays[frame_idx] == 0:
                closest = 'resting'
                top_n = ['resting'] * 5
                top_n_similarities = [0] * 5  # Assuming similarity is 0 for resting

        # Collect predictions
        if gloss_hand is None:
            try:
                similarity_ratio = top_n_similarities[0] / top_n_similarities[1]
            except (IndexError, ZeroDivisionError):
                similarity_ratio = 0
            similarities.append(similarity_ratio)
            pred_top1.append(closest)
            pred_top3.extend(top_n[:3])
            pred_top3_list.append(top_n[:3])
            pred_top5.extend(top_n[:5])
            idxs_list.append(idx)

            # Update predictions dictionary for plotting
            predictions['closest'].append(closest)
            predictions['top_n'].append(top_n)
        else:
            # Ensure gloss_hand lists have sufficient length
            if frame_idx < len(handedness):
                current_handedness = handedness[frame_idx]
            else:
                current_handedness = 'Unknown'

            if frame_idx < len(weak):
                current_weak = weak[frame_idx]
            else:
                current_weak = 'Unknown'

            if frame_idx < len(strong):
                current_strong = strong[frame_idx]
            else:
                current_strong = 'Unknown'

            pred_top1.append(closest)
            pred_top3.extend(top_n[:3])
            pred_top5.extend(top_n[:5])
            idxs_list.append(idx)

            # Update predictions dictionary for plotting
            predictions['closest'].append(closest)
            predictions['top_n'].append(top_n)
            predictions['handedness'].append(current_handedness)
            predictions['weak'].append(current_weak)
            predictions['strong'].append(current_strong)

        # Note: Plotting is handled separately

    # After processing all frames, handle plotting
    if plot:
        plot_pose_3d(
            keypoints=keypoints,
            predictions=predictions,
            output_path=output_gif_path,
            filename=filename,
            gloss_hand=gloss_hand
        )

    # Aggregate predictions using Counters
    counter_top1 = Counter(pred_top1)
    counter_top3 = Counter(pred_top3)
    most_common_top3 = counter_top3.most_common(3)
    counter_top5 = Counter(pred_top5)
    most_common_top5 = counter_top5.most_common(5)

    # Return results based on the presence of ground truth
    if gloss_hand is None:
        return counter_top1, counter_top3, counter_top5, pred_top1, pred_top3_list, similarities
    else:
        # Determine the most common index if needed
        idx = Counter(idxs_list).most_common(1)[0][0] if idxs_list else None
        median_top1 = counter_top1.most_common(1)[0][0] if counter_top1 else None
        median_top3 = [item[0] for item in most_common_top3]
        median_top5 = [item[0] for item in most_common_top5]
        # Assuming you want to return the last top_n_similarities
        last_top_n_similarities = top_n_similarities if 'top_n_similarities' in locals() else None
        return median_top1, median_top3, median_top5, idx, last_top_n_similarities


def load_and_predict(data, base_filename, input_folder, output_folder, reference_poses, gloss_hand, booleans = None, transformation = 'pdm'):
    """
    Loads left and right pose files, performs prediction, and returns the results.

    Args:
        base_filename (str): Base filename without suffix.
        input_folder (str): Directory containing input .pkl files.
        output_folder (str): Directory to save output GIFs.
        reference_poses (dict): Reference poses for comparison.
        gloss_hand (dict): Ground truth gloss hand information.

    Returns:
        dict: Predictions and poses for both hands.
    """
    predictions = {}
    top1 = {}
    top_3 = {}
    poses = {}
    idxs = {}

    def set_resting_hand(predictions, top1 , top_3, poses, suffix, boolean_arrays):
        print(suffix[1], ' hand inactive')
        predicted_class = ['resting' for i in range(len(boolean_arrays))]
        predictions[suffix] = (predicted_class, predicted_class, predicted_class)
        top1[suffix] = predicted_class
        top_3[suffix] = []
        poses[suffix] = None
        return predicted_class, predictions, top1, top_3, poses

    for suffix in ['-R.pkl', '-L.pkl']:
        hand = suffix[1]
        
        if hand == 'L': idx = 0
        elif hand == 'R': idx = 1

        if suffix == '-L.pkl':
            if np.sum(booleans[idx]) == 0:
                predicted_class, predictions, top1, top_3, poses = set_resting_hand(predictions, top1 , top_3, poses, suffix, booleans[idx])
                continue

        file_path = os.path.join(input_folder, base_filename + suffix)
        if not os.path.exists(file_path):

            print(f"File not found: {file_path}")
            continue

        print(f"Processing file: {file_path}")
        
        if gloss_hand is None: handshape = base_filename
        else: handshape = gloss_hand['handshape']

        output_gif_path = os.path.join(
            output_folder,
            f"{base_filename}_{handshape}_{suffix[1:2]}.gif"
        )

        if hand == 'L':
            pose = data.normalized_hamer_left
        elif hand == 'R':
            pose = data.normalized_hamer_right

        predicted_class, top3, top5, pred_top1, pred_top3, similarity = predict_handshape(
            keypoints=pose,
            reference_poses=reference_poses,
            output_gif_path=output_gif_path,
            filename=base_filename,
            gloss_hand=gloss_hand,
            plot=False,
            boolean_arrays = booleans[idx],
            transformation=transformation
        )
  
        predictions[suffix] = (predicted_class, top3, top5)
        poses[suffix] = pose
        top1[suffix] = pred_top1
        top_3[suffix] = pred_top3

    return predictions, poses, idxs, top1, top_3, similarity



def update_counters(side, pred, top3, top5, ground_truth, counts, base_filename, handedness, strong=None, weak=None):
    """
    Updates the correct and total prediction counters.

    Args:
        side (str): 'L' or 'R'.
        pred (str): Predicted class.
        top3 (list): Top 3 predicted classes.
        top5 (list): Top 5 predicted classes.
        ground_truth (str): Ground truth class.
        counts (dict): Dictionary to hold correct and total counts.
        base_filename (str): Base filename being processed.
        handedness (str): Handedness type.
        strong (str, optional): Strong handshape. Defaults to None.
        weak (str, optional): Weak handshape. Defaults to None.

    Returns:
        None
    """
    counts['true'].append(ground_truth)
    counts['pred'].append(pred)

    if ground_truth != pred:
        print('-----------------------------')
        print(f"Processing file: {base_filename}, handedness: {handedness}")
        if handedness == '2a':
            print(f"Strong hand: {strong}, Weak hand: {weak}")
        print(f"Top 3 most common predictions: {top3[:3]}")
        print(f"Top 5 most common predictions: {top5[:5]}")
        print(f"File: {side} - Predicted Handshape: {pred} - Ground Truth Handshape: {ground_truth}")

    if pred == ground_truth:
        counts['correct'] += 1
        counts['class_correct'][ground_truth] = counts['class_correct'].get(ground_truth, 0) + 1
    if ground_truth in top3:
        counts['top3_correct'] += 1
    if ground_truth in top5:
        counts['top5_correct'] += 1
    counts['class_total'][ground_truth] = counts['class_total'].get(ground_truth, 0) + 1


def evaluate_predictions(gloss_hand, predictions, counts, reference_poses, base_filename, idxs):
    """
    Evaluates predictions based on handedness and updates counters accordingly.

    Args:
        gloss_hand (dict): Ground truth gloss hand information.
        predictions (dict): Predictions for '-L.pkl' and '-R.pkl'.
        counts (dict): Dictionary to hold counts and labels.
        reference_poses (dict): Reference poses for comparison.
        base_filename (str): Base filename being processed.
        idxs (dict): Indices for reference poses.

    Returns:
        None
    """
    strong = gloss_hand['handshape']
    handedness = gloss_hand['handedness']
    weak = gloss_hand['weak']

    if handedness == '2s':
        # Strong handedness: both hands are strong
        for suffix, side in [('-L.pkl', 'L'), ('-R.pkl', 'R')]:
            if suffix in predictions:
                pred, top3, top5 = predictions[suffix]
                update_counters(
                    side=side,
                    pred=pred,
                    top3=top3,
                    top5=top5,
                    ground_truth=strong,
                    counts=counts,
                    base_filename=base_filename,
                    handedness=handedness
                )

    elif handedness == '1':
        # Single handedness: determine which hand to consider
        if strong not in reference_poses:
            print(f"Strong pose not found for handshape: {strong}")
            return

        selected_hand = select_hand(
            pose_L=counts['poses'].get('-L.pkl'),
            pose_R=counts['poses'].get('-R.pkl'),
            strong_pose=reference_poses.get(strong)
        )

        if selected_hand == 'L' and '-L.pkl' in predictions:
            pred, top3, top5 = predictions['-L.pkl']
            update_counters(
                side='L',
                pred=pred,
                top3=top3,
                top5=top5,
                ground_truth=strong,
                counts=counts,
                base_filename=base_filename,
                handedness=handedness
            )
        elif selected_hand == 'R' and '-R.pkl' in predictions:
            pred, top3, top5 = predictions['-R.pkl']
            update_counters(
                side='R',
                pred=pred,
                top3=top3,
                top5=top5,
                ground_truth=strong,
                counts=counts,
                base_filename=base_filename,
                handedness=handedness
            )
        else:
            print(f"Something went wrong with hand selection for file: {base_filename}")
            exit()

    elif handedness == '2a':
        # Dual handedness: strong and weak
        if strong == weak:
            # Both hands have the same handshape
            for suffix, side in [('-L.pkl', 'L'), ('-R.pkl', 'R')]:
                if suffix in predictions:
                    pred, top3, top5 = predictions[suffix]
                    update_counters(
                        side=side,
                        pred=pred,
                        top3=top3,
                        top5=top5,
                        ground_truth=strong,
                        counts=counts,
                        base_filename=base_filename,
                        handedness=handedness,
                        strong=strong,
                        weak=weak
                    )
        else:
            # Different handshapes for each hand
            if weak not in reference_poses:
                print(f"Weak pose not found for handshape: {weak}")
                return

            strong_pose = reference_poses.get(strong)
            weak_pose = reference_poses.get(weak)

            hands = select_hand(
                pose_L=counts['poses'].get('-L.pkl'),
                pose_R=counts['poses'].get('-R.pkl'),
                strong_pose=strong_pose,
                weak_pose=weak_pose,
                handedness=handedness,
                idx_L=idxs.get('L', 0),
                idx_R=idxs.get('R', 0)
            )

            for suffix, side in [('-L.pkl', 'L'), ('-R.pkl', 'R')]:
                if suffix in predictions:
                    pred, top3, top5 = predictions[suffix]
                    if hands[side] == 'strong':
                        gt_hs = strong
                    elif hands[side] == 'weak':
                        gt_hs = weak
                    else:
                        print(f"Unknown hand strength for side {side} in file {base_filename}")
                        continue
                    update_counters(
                        side=side,
                        pred=pred,
                        top3=top3,
                        top5=top5,
                        ground_truth=gt_hs,
                        counts=counts,
                        base_filename=base_filename,
                        handedness=handedness,
                        strong=strong,
                        weak=weak
                    )


def process_directory(input_folder, output_folder, ground_truth_labels, reference_poses, device):
    """
    Processes all .pkl files in the input directory, predicts handshapes, and evaluates accuracy.

    Args:
        input_folder (str): Directory containing .pkl pose data files.
        output_folder (str): Directory to save output GIFs.
        ground_truth_labels (dict): Ground truth labels.
        reference_poses (dict): Reference poses for comparison.
        device (torch.device): Computation device.
    """
    counts = {
        'correct': 0,
        'top3_correct': 0,
        'top5_correct': 0,  # Added for top-5 accuracy
        'class_correct': {},
        'class_total': {},
        'true': [],
        'pred': [],
        'poses': {}
    }
    prediction_arrays = {}
    total_files = 0

    # Collect base filenames without suffixes
    base_filenames = set()
    for filename in os.listdir(input_folder):
        if filename.endswith('.pkl'):
            base_filename = filename[:-6]
            base_filenames.add(base_filename)

    base_filenames = sorted(base_filenames)
    for base_filename in base_filenames:
        gloss_hand = ground_truth_labels.get(base_filename)
        if not gloss_hand or gloss_hand['hs_change'] != '-1':
            continue

        predictions, poses, idxs, pred_top1, similarity = load_and_predict(
            base_filename=base_filename,
            input_folder=input_folder,
            output_folder=output_folder,
            reference_poses=reference_poses,
            gloss_hand=gloss_hand, 
            transformation=transformation
        )
        prediction_arrays[base_filename] = pred_top1


        if not predictions:
            continue

        counts['poses'] = poses
        try:
            evaluate_predictions(
                gloss_hand=gloss_hand,
                predictions=predictions,
                counts=counts,
                reference_poses=reference_poses,
                base_filename=base_filename,
                idxs=idxs
            )
            if gloss_hand['handedness'] == '1':
                total_files += 1
            else:
                total_files += 2  # Both L and R are processed
        except Exception as e:
            print(e)
            continue
    print("\nConfusion Matrix:")
    plot_cm(counts['true'], counts['pred'])

    # Calculate and display overall accuracy
    accuracy = (counts['correct'] / total_files) * 100 if total_files > 0 else 0
    accuracy_top3 = (counts['top3_correct'] / total_files) * 100 if total_files > 0 else 0
    accuracy_top5 = (counts['top5_correct'] / total_files) * 100 if total_files > 0 else 0  # Added top-5 accuracy
    print(f"\nTotal Accuracy: {accuracy:.2f}%")
    print(f"Total Accuracy Top-3: {accuracy_top3:.2f}%")
    print(f"Total Accuracy Top-5: {accuracy_top5:.2f}%")  # Print top-5 accuracy

    # Calculate and display class-wise accuracy
    print("\nClass-Distributed Accuracy:")
    for handshape_class, correct_count in counts['class_correct'].items():
        total_count = counts['class_total'].get(handshape_class, 0)
        class_accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
        print(f"Class {handshape_class}: {class_accuracy:.2f}% accuracy ({correct_count}/{total_count})")

    print('\nClass Totals:')
    for handshape_class, total_count in counts['class_total'].items():
        print(f"Class {handshape_class}: {total_count}")


def inference_directory(data, input_folder, output_folder, reference_poses, boolean_arrays, base_filename=None, transformation = 'pdm'):
    """
    Processes all .pkl files in the input directory, predicts handshapes, and evaluates accuracy.

    Args:
        input_folder (str): Directory containing .pkl pose data files.
        output_folder (str): Directory to save output GIFs.
        ground_truth_labels (dict): Ground truth labels.
        reference_poses (dict): Reference poses for comparison.
        device (torch.device): Computation device.
    """
    predicted_arrays = {}
    predicted_arrays_3 = {}

        
    predictions, poses, idxs, pred_top1, pred_top3, similarity = load_and_predict(
        data, base_filename=base_filename,
        input_folder=input_folder,
        output_folder=output_folder,
        reference_poses=reference_poses,
        gloss_hand=None,
        booleans=boolean_arrays,
        transformation=transformation
    )

    for item in pred_top1:
        if 'normalized_' in base_filename:
            base_filename = base_filename.replace('normalized_', '')
        predicted_arrays[base_filename+item] = pred_top1[item]
        
    for item in pred_top3:
        if 'normalized_' in base_filename:
            base_filename = base_filename.replace('normalized_', '')    
        predicted_arrays_3[base_filename+item] = pred_top3[item]
    
    return predicted_arrays, predicted_arrays_3, similarity 




def main(args):
    """
    Main function to execute the handshape prediction and evaluation.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load ground truth labels
    ground_truth_labels = read_ground_truth_labels(args.sign_list_file)

    # Ensure output directory exists
    os.makedirs(args.output_folder, exist_ok=True)

    # Process all .pkl files in the input directory
    process_directory(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        ground_truth_labels=ground_truth_labels,
        reference_poses=reference_poses,
        device=device
    )



def main_handshape(data, input_folder, output_folder, boolean_arrays = None, base_filename = None, transformation = 'pdm'):
    reference_poses = data.reference_poses
   
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Process all .pkl files in the input directory

    predictions = inference_directory(
        data, 
        input_folder=input_folder,
        output_folder=output_folder,
        reference_poses=reference_poses,
        boolean_arrays=boolean_arrays,
        base_filename=base_filename,
        transformation=transformation
    )
    return predictions
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handshape Prediction and Evaluation Script")
    parser.add_argument(
        '--input_folder',
        type=str,
        default='/home/gomer/oline/PoseTools/data/datasets/test_data/segmented_hamer_pkl',
        #default='../../../../mnt/fishbowl/gomer/oline/hamer_pkl',
        help='Directory containing .pkl pose data files'
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        default='/home/gomer/oline/PoseTools/data/datasets/test_data/gifs/',
        help='Directory to save output GIFs'
    )
    parser.add_argument(
        '--sign_list_file',
        type=str,
        default= '/home/gomer/oline/PoseTools/data/datasets/test_data/sign_list.txt', #'/home/gomer/oline/PoseTools/data/metadata/sign_lists/SB_list.txt',  #
        help='File containing ground truth labels with handedness'
    )
    parser.add_argument(
        '--reference_pose_path',
        type=str,
        default='/home/gomer/oline/PoseTools/src/modules/handshapes/utils/references/reference_poses_pdm_extended_uva.pkl',
        help='File containing reference poses'
    )
    parser.add_argument(
        '--reference_pose_path_sb',
        type=str,
        default='/home/gomer/oline/PoseTools/src/modules/handshapes/utils/references/reference_poses_pdm_extended_pluss.pkl',
        help='File containing reference poses'
    )
    args = parser.parse_args()

    gloss_mapping = read_dict_from_txt('/home/gomer/oline/PoseTools/data/metadata/output/global_value_to_id.txt')
    # Load reference poses
    with open(args.reference_pose_path, 'rb') as file:
        reference_poses = pickle.load(file)

    with open(args.reference_pose_path_sb, 'rb') as file:
        reference_poses_sb = pickle.load(file)
    
    #reference_poses['5r'] = reference_poses_sb['5r']
    print(f"Reference poses loaded: {list(reference_poses.keys())}")
    print(f"Number of reference poses: {len(list(reference_poses.keys()))}")
    
    

    main(args)
