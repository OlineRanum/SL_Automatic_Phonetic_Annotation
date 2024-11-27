import cv2
import requests
import json
import sys
import tempfile
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import butter, filtfilt
import subprocess
import sys
import argparse
import os
import torch
import omegaconf

# Define the Joint class
class Joint:
    def __init__(self, idx, parent_idx, angle, distance):
        self.idx = idx
        self.parent_idx = parent_idx
        self.angle = angle
        self.distance = distance
        self.world_coords = None

# Define the ManoForwardKinematics class
class ManoForwardKinematics:
    def __init__(self):
        # Define the connections between joints (tuples of indices)
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index Finger
            (0, 9), (9, 10), (10, 11), (11, 12), # Middle Finger
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring Finger
            (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky Finger
        ]

    def get_angles(self, direction):
        pitch = np.arcsin(direction[2])
        yaw = np.arctan2(direction[1], direction[0])
        return yaw, pitch

    def normalize_vector(self, vector):
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def get_distance(self, origin, target):
        return np.linalg.norm(target - origin)

    def calculate_new_position(self, origin, yaw, pitch, distance):
        x0, y0, z0 = origin
        x = x0 + distance * np.cos(pitch) * np.cos(yaw)
        y = y0 + distance * np.cos(pitch) * np.sin(yaw)
        z = z0 + distance * np.sin(pitch)
        return np.array([x, y, z])

    def joint_to_world(self, target_joint, parent_world_pos):
        return self.calculate_new_position(parent_world_pos, target_joint.angle[0], target_joint.angle[1],
                                           target_joint.distance)

    def forward_kinematics(self, keypoints):
        forward_kinematics = {}
        # Build the forward kinematics data
        for connection in self.connections:
            parent_id = connection[0]
            target_id = connection[1]
            origin = np.array(keypoints[parent_id])
            target = np.array(keypoints[target_id])

            direction = self.normalize_vector(target - origin)
            angles = self.get_angles(direction)
            distance = self.get_distance(origin, target)
            forward_kinematics[target_id] = Joint(target_id, parent_id, angles, distance)

        # Initialize the root joint
        root_joint = Joint(0, 0, (0, 0), 0)
        root_joint.world_coords = np.array(keypoints[0])
        forward_kinematics[0] = root_joint

        # Compute world coordinates for each joint
        world_coordinates = {0: root_joint.world_coords}
        for joint_index, joint in forward_kinematics.items():
            if joint_index == 0:
                continue
            parent_world_pos = world_coordinates[joint.parent_idx]
            joint.world_coords = self.joint_to_world(joint, parent_world_pos)
            world_coordinates[joint.idx] = joint.world_coords

        return forward_kinematics

    def apply_fixed_perspective_transformation(self, landmarks):
        # Step 1: Define key points
        wrist_index = 0  # Assuming wrist is at index 0
        leftmost_index = 17  # Pinky finger MCP
        rightmost_index = 5  # Index finger MCP

        wrist = landmarks[wrist_index]
        leftmost = landmarks[leftmost_index]
        rightmost = landmarks[rightmost_index]

        # Step 2: Compute vectors for the new coordinate system
        u = rightmost - leftmost  # x-axis vector
        v = wrist - leftmost      # vector from leftmost to wrist

        # Step 3: Compute normal vector to the palm plane (z-axis)
        n = np.cross(u, v)
        if np.linalg.norm(n) == 0:
            n = np.array([0, 0, 1])  # Default normal vector if cross product is zero

        # Step 4: Compute the new y-axis vector (perpendicular to x and z axes)
        v_new = np.cross(n, u)

        # Step 5: Normalize the axes to create an orthonormal basis
        u_unit = self.normalize_vector(u)
        v_unit = self.normalize_vector(v_new)
        n_unit = self.normalize_vector(n)

        # Step 6: Create the rotation matrix from the new basis vectors
        rotation_matrix = np.stack((u_unit, v_unit, n_unit), axis=1)  # Shape: (3, 3)

        # Step 7: Translate landmarks so that 'leftmost' becomes the origin
        landmarks_centered = landmarks - leftmost

        # Step 8: Apply the rotation to align the hand to the new coordinate system
        landmarks_aligned = landmarks_centered @ rotation_matrix

        return landmarks_aligned

    def butterworth_filter(self, data, cutoff, fs, order=5):
        # Apply Butterworth filter to the data
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        # Apply the filter along the frames (axis=0)
        y = filtfilt(b, a, data, axis=0, padlen=0)
        return y

    def process_hand_data(self, hand_data, hand_label, output_dir, hamer_file, create_gif=False):
        # Collect all frames' landmarks
        frames_landmarks = []

        # First, extract landmarks from each frame

        for frame in hand_data:
            if not frame or not frame[0]:
                frames_landmarks.append(None)
                continue

            landmarks = np.array(frame[0])

            # Ensure landmarks are in the correct shape
            if landmarks.shape[0] != 21:
                print(f"Warning: Expected 21 landmarks, but got {landmarks.shape[0]} for {hand_label}")
                frames_landmarks.append(None)
                continue

            # Adjust the indices if necessary (here, index 20 might be the wrist)
            if np.all(landmarks[20] == [0.0, 0.0, 0.0]) or np.allclose(landmarks[20], [0.0, 0.0, 0.0]):
                # Move the last point to the first position
                landmarks = np.vstack([landmarks[20], landmarks[:20]])

            frames_landmarks.append(landmarks)

        # Now, apply the Butterworth filter to the collected landmarks
        # We need to handle missing frames (None values)
        valid_frames = [frame for frame in frames_landmarks if frame is not None]
        if not valid_frames:
            print(f"No valid frames for {hand_label}")
            return None, None

        # Stack the frames to create a 3D array (num_frames x num_landmarks x 3)
        data_array = np.stack(valid_frames)
        num_frames = data_array.shape[0]
        fs = 30  # Sampling frequency (assuming 30 frames per second)
        cutoff = 3  # Desired cutoff frequency of the filter, in Hz

        # Apply the filter along the frames axis
        try:
            filtered_data_butterworth = self.butterworth_filter(data_array, cutoff, fs, order=3)
            print(filtered_data)
            print(filtered_data_butterworth)    
            filtered_data = filtered_data_butterworth
        except ValueError as e:
            print(f"Error filtering {hand_label} data: {e}")
            return None, None

        # Now, replace the landmarks in frames_landmarks with the filtered data
        filtered_frames_landmarks = []
        j = 0  # Index for valid frames
        for i in range(len(frames_landmarks)):
            if frames_landmarks[i] is not None:
                filtered_frames_landmarks.append(filtered_data[j])
                j += 1
            else:
                filtered_frames_landmarks.append(None)

        # If visualization is desired, generate GIF
        if create_gif:
            self.generate_gif(filtered_frames_landmarks, hand_label, output_dir, hamer_file)

        # Return the filtered frames
        print(filtered_frames_landmarks)
        return filtered_frames_landmarks, valid_frames

    def normalize_and_save(self, hand_data, hand_label, hamer_file, output_path, create_gif=False):
        # Process hand data and apply filtering
        filtered_frames, valid_frames = self.process_hand_data(hand_data, hand_label, output_path, hamer_file, create_gif)
        
        if filtered_frames is None:
            return None

        # Prepare the data for JSON output
        output_frames = []
        for frame in filtered_frames:
            if frame is None:
                output_frames.append([])
            else:
                # Process landmarks for fixed perspective transformation
                forward_kinematics = self.forward_kinematics(frame)

                # Reconstruct landmarks from forward kinematics
                reconstructed_landmarks = np.zeros_like(frame)
                for idx, joint in forward_kinematics.items():
                    reconstructed_landmarks[idx] = joint.world_coords

                # Apply fixed perspective transformation
                transformed_landmarks = self.apply_fixed_perspective_transformation(reconstructed_landmarks)

                # Convert the transformed landmarks to list for JSON output
                output_frames.append(transformed_landmarks.tolist())
        print(len(hand_data))
        return output_frames
