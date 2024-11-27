# pose_video_analyzer.py

import os
import pickle
import numpy as np
from hmmlearn import hmm
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.signal import medfilt  # For signal smoothing
from itertools import groupby
from operator import itemgetter


class ActivationAnalyzer:
    def __init__(self, data, BASE_DIR=None, n_components = 2):
        self.data = data
        self.n_components = n_components
        
        self.pose = data.pose
        self.hamer_right = data.normalized_hamer_right
        self.hamer_left = data.normalized_hamer_left
        
        self.left_wrist = data.wrist_left
        self.right_wrist = data.wrist_right

        self.num_frames = data.num_frames

        self.normal_vectors_right = data.normal_vectors_right
        self.normal_vectors_left = data.normal_vectors_left
        print(len(self.left_wrist))

    def train_hmm_with_best_initialization(
        self,
        features,
        covariance_type="diag",
        n_iter=300,
        tol=1e-3,
        n_initializations=5
    ):
        best_score = -np.inf
        best_model = None

        for i in range(n_initializations):
            try:
                model = hmm.GaussianHMM(
                    n_components=self.n_components,  # Use self.n_components
                    covariance_type=covariance_type,
                    n_iter=n_iter,
                    tol=tol,
                    random_state=i,
                    verbose=False
                )
                model.fit(features)
                score = model.score(features)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                print(f"Initialization {i} failed: {e}")

        if best_model:
            return best_model
        else:
            raise ValueError("All HMM initializations failed.")

    def interpret_states(self, model, hidden_states):
        """
        Interpret hidden states into movement labels based on the number of components.
        For 2 components: Active and Inactive.
        For 3 components: Active, Transition, and Inactive.
        """
        state_means = model.means_.mean(axis=1)
        sorted_states = np.argsort(state_means)

        if self.n_components == 2:
            # Map the states based on emission means
            movement_state = np.argmax(state_means)
            movement_labels = np.where(hidden_states == movement_state, 'Active', 'Inactive')
        elif self.n_components == 3:
            # Map the states to 'Inactive', 'Transition', 'Active'
            state_mapping = {}
            state_mapping[sorted_states[0]] = 'Inactive'
            state_mapping[sorted_states[1]] = 'Transition'
            state_mapping[sorted_states[2]] = 'Active'

            movement_labels = np.array([state_mapping[state] for state in hidden_states])
        else:
            raise ValueError("Unsupported number of HMM components.")

        return movement_labels

    def find_movement_segments(self, movement_labels):
        """
        Identify the start and end frames of Active segments.
        Handles both 2 and 3 states.
        """
        if self.n_components == 2:
            # For 2 states, labels are 'Active' and 'Inactive'
            active_label = 'Active'
        elif self.n_components == 3:
            # For 3 states, labels are 'Active', 'Transition', 'Inactive'
            active_label = 'Active'  # You can also consider 'Transition' if needed
        else:
            raise ValueError("Unsupported number of HMM components.")

        active_indices = np.where(movement_labels == active_label)[0]
        if active_indices.size == 0:
            return None, None

        # Find contiguous regions of Active states
        groups = []
        for k, g in groupby(enumerate(active_indices), lambda ix: ix[0] - ix[1]):
            group = list(map(itemgetter(1), g))
            groups.append((group[0], group[-1]))

        # For simplicity, take the first and last Active segments
        start_frame = groups[0][0]
        end_frame = groups[-1][1]

        return start_frame, end_frame

    
    def smooth_state_array(self, state_array, window_size=3):
        """
        Smooth the state array by taking the median in a sliding window.
        """
        if self.n_components == 2:
            state_encoding = {'Inactive': 0, 'Active': 1}
        elif self.n_components == 3:
            state_encoding = {'Inactive': 0, 'Transition': 1, 'Active': 2}
        else:
            raise ValueError("Unsupported number of HMM components.")

        # Encode states as integers
        encoded = np.array([state_encoding[state] for state in state_array])
        
        # Apply median filter
        smoothed = medfilt(encoded, kernel_size=window_size)
        
        # Decode back to state labels
        decoding = {v: k for k, v in state_encoding.items()}
        smoothed_states = np.array([decoding[val] for val in smoothed])
        
        return smoothed_states


    def detect_movement_frames(self, plot=True):
        # Preprocessing steps (unchanged)
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        left_wrist = np.expand_dims(self.left_wrist, 1)
        right_wrist = np.expand_dims(self.right_wrist, 1)
        left_wrist_vel = np.diff(left_wrist, axis=0)  # Shape: [T-1, dims]
        left_wrist_vel = np.vstack((left_wrist_vel, np.zeros((1, left_wrist_vel.shape[1]))))  
        right_wrist_vel = np.diff(right_wrist, axis=0)  # Shape: [T-1, dims]
        right_wrist_vel = np.vstack((right_wrist_vel, np.zeros((1, right_wrist_vel.shape[1]))))

        # PCA on handshape features
        pca = PCA(n_components=1)
        if len(self.hamer_left) < len(self.pose):
            diff = len(self.pose) - len(self.hamer_left)
            self.hamer_left = np.pad(
                self.hamer_left, 
                pad_width=((0, diff), (0, 0), (0, 0)), 
                mode='constant', 
                constant_values=0
            )
        if len(self.hamer_right) < len(self.pose):
            diff = len(self.pose) - len(self.hamer_right)
            self.hamer_right = np.pad(
                    self.hamer_right, 
                    pad_width=((0, diff), (0, 0), (0, 0)), 
                    mode='constant', 
                    constant_values=0
                )
        print('Hammer left shape', self.hamer_left.shape)
        print('Hammer right shape', self.hamer_right.shape)
        handshape_left_flat = self.hamer_left.reshape(self.num_frames, -1)
        handshape_right_flat = self.hamer_right.reshape(self.num_frames, -1)

        handshape_features_left = np.abs(pca.fit_transform(handshape_left_flat))
        handshape_features_right = np.abs(pca.fit_transform(handshape_right_flat))

        # Scale PCA features
        handshape_left_scaled = scaler.fit_transform(handshape_features_left)
        handshape_right_scaled = scaler.fit_transform(handshape_features_right)

        # scale the normal vectors
        self.normal_vectors_left = scaler.fit_transform(self.normal_vectors_left)
        self.normal_vectors_right = scaler.fit_transform(self.normal_vectors_right)
        

        # Combine features
        oxl,oyl,ozl = [[x] for x in self.normal_vectors_left[:,0]], [[y] for y in self.normal_vectors_left[:,1]], [[z] for z in self.normal_vectors_left[:,2]]
        oxr,oyr,ozr = [[x] for x in self.normal_vectors_right[:,0]], [[y] for y in self.normal_vectors_right[:,1]], [[z] for z in self.normal_vectors_right[:,2]]
        combined_features_left = np.hstack((handshape_left_scaled, left_wrist, left_wrist_vel))#, self.normal_vectors_left[:,1], self.normal_vectors_left[:,2]))
        combined_features_right = np.hstack((handshape_right_scaled, right_wrist, right_wrist_vel))#, self.normal_vectors_right[:,1], self.normal_vectors_right[:,2]))
        joint_features = np.vstack((combined_features_left, combined_features_right))   
        
        # Train HMM
        model = self.train_hmm_with_best_initialization(
            joint_features,
            covariance_type="diag",
            n_iter=300,
            tol=1e-1,
            n_initializations=5
        )
        
        hidden_states = model.predict(joint_features)
        
        # Interpret states
        movement_labels = self.interpret_states(model, hidden_states)
        
        # Split movement labels back into left and right
        movement_labels_left = movement_labels[:self.num_frames]
        movement_labels_right = movement_labels[self.num_frames:]
        
        # Smooth the state sequences
        self.movement_labels_left = self.smooth_state_array(movement_labels_left)
        self.movement_labels_right = self.smooth_state_array(movement_labels_right)
        
        # Identify movement segments
        left_start_frame, left_end_frame = self.find_movement_segments(self.movement_labels_left)
        right_start_frame, right_end_frame = self.find_movement_segments(self.movement_labels_right)

        # Print results
        if left_start_frame is None:
            print("No movement detected in left hand")
        else:
            print(f"\nLeft hand movement starts at frame {left_start_frame//3}")
            print(f"Left hand movement ends at frame {left_end_frame//3}\n")

        if right_start_frame is None:
            print("No movement detected in right hand")
        else:
            print(f"Right hand movement starts at frame {right_start_frame//3}")
            print(f"Right hand movement ends at frame {right_end_frame//3}")

        # Plot hidden states
        if plot:
            self.plot_hidden_states(self.movement_labels_left, self.movement_labels_right)

        # Return frames
        
        start_frame = [left_start_frame, right_start_frame]
        end_frame = [left_end_frame, right_end_frame]

        return start_frame, end_frame


    def plot_hidden_states(self, movement_labels_left, movement_labels_right):
        movement_labels_left = movement_labels_left[::3]
        movement_labels_right = movement_labels_right[::3]
        frames = np.arange(len(movement_labels_left))

        if self.n_components == 2:
            state_colors = {'Inactive': 'grey', 'Active': 'blue'}
        elif self.n_components == 3:
            state_colors = {'Inactive': 'grey', 'Transition': 'yellow', 'Active': 'blue'}
        else:
            raise ValueError("Unsupported number of HMM components.")

        plt.figure(figsize=(15, 8))

        # Plot Left Hand
        plt.subplot(2, 1, 1)
        for state, color in state_colors.items():
            plt.plot(frames, (movement_labels_left == state).astype(int), label=f'Left {state}', color=color)
            plt.fill_between(frames, 0, (movement_labels_left == state).astype(int), where=(movement_labels_left == state), color=color, alpha=0.3)
        plt.xlabel('Frame')
        plt.ylabel('State')
        plt.title('Left Hand Movement Detection')
        plt.legend()

        # Plot Right Hand
        plt.subplot(2, 1, 2)
        for state, color in state_colors.items():
            plt.plot(frames, (movement_labels_right == state).astype(int), label=f'Right {state}', color=color)
            plt.fill_between(frames, 0, (movement_labels_right == state).astype(int), where=(movement_labels_right == state), color=color, alpha=0.3)
        plt.xlabel('Frame')
        plt.ylabel('State')
        plt.title('Right Hand Movement Detection')
        plt.legend()

        plt.tight_layout()
        plt.show()


    def find_movement_segments(self, movement_labels):
        """
        Identify the start and end frames of Active segments.
        Handles both 2 and 3 states.
        """
        if self.n_components == 2:
            # For 2 states, labels are 'Active' and 'Inactive'
            active_label = 'Active'
        elif self.n_components == 3:
            # For 3 states, labels are 'Active', 'Transition', 'Inactive'
            active_label = 'Active'  # You can also consider 'Transition' if needed
        else:
            raise ValueError("Unsupported number of HMM components.")

        active_indices = np.where(movement_labels == active_label)[0]
        if active_indices.size == 0:
            return None, None

        # Find contiguous regions of Active states
        groups = []
        for k, g in groupby(enumerate(active_indices), lambda ix: ix[0] - ix[1]):
            group = list(map(itemgetter(1), g))
            groups.append((group[0], group[-1]))

        # For simplicity, take the first and last Active segments
        start_frame = groups[0][0]
        end_frame = groups[-1][1]

        return start_frame, end_frame

    def construct_boolean(self, pose):
        """
        Construct boolean arrays for movement labels, adaptable for two or three states.

        Parameters:
        - pose (np.ndarray): Pose data.
        - first (list): List of starting indices for left and right hands.
        - last (list): List of ending indices for left and right hands.
        - mode (str): 'full' to return the movement labels, 'start_stop' for boolean arrays.
        - n_states (int): Number of states (2 or 3).

        Returns:
        - tuple: Two numpy arrays for left and right hand movement.
        """
        bool_L = np.zeros(len(pose))
        bool_R = np.zeros(len(pose))

        for i in range(len(bool_L)):
            if self.movement_labels_left[i] == 'Active':
                bool_L[i] = 1
            elif self.movement_labels_left[i] == 'Inactive':
                bool_L[i] = 0
            elif self.movement_labels_left[i] == 'Transition':
                bool_L[i] = 0
            if self.movement_labels_right[i] == 'Active':
                bool_R[i] = 1
            elif self.movement_labels_right[i] == 'Inactive':
                bool_R[i] = 0
            elif self.movement_labels_right[i] == 'Transition':
                bool_R[i] = 0
                
        return [bool_L, bool_R]



def main_activation(data, n_components = 2):
    analyzer = ActivationAnalyzer(data, n_components=n_components)

    first, last = analyzer.detect_movement_frames()
    
    # TODO: This is not fixed yet 
    boolean_activation_labels = analyzer.construct_boolean(analyzer.pose)
    activation_labels = [analyzer.movement_labels_left, analyzer.movement_labels_right]
    
    first = [v for v in first if v is not None]
    last = [v for v in last if v is not None]

    return boolean_activation_labels, activation_labels, np.min(first), np.max(last) 

if __name__ == "__main__":
    base_filename = "M20241107_6254"
    gif_path = '/home/gomer/oline/PoseTools/src/modules/demo/graphics/gifs/wrist_movement_animation.gif'
    main_activation(base_filename, create_anim=True, save_anim_path=gif_path)
