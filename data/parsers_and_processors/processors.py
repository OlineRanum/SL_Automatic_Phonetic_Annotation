from pose_format import Pose
import numpy.ma as ma
import numpy as np
import os

import pickle


class PoseFormatProssesor:
    def __init__(self, path="A.pose"):
        self.pose_path = path

    def preprocess_pose(self, feat, conf):
        # Trim 1/3 of the video from the start and the end
        n_frames = feat.shape[0]  
        trim_start = n_frames // 3  # Calculate the start index for trimming
        trim_end = 2 * n_frames // 3  # Calculate the end index for trimming

        feat = feat[trim_start:trim_end,:, :, :]
        conf = conf[trim_start:trim_end, :, :]

        # get the right hand
        feat = feat[:, :, :543, :]
        feat = feat[:, :, -21:, :].squeeze(1)

        conf = conf[:, :, :543]
        conf = conf[:, :, -21:]

        # Normalize size
        flat_array = feat.reshape(-1, 3)
          
        # Find the min and max for x and y coordinates
        min_vals = np.min(flat_array, axis=0)  # Min of x and y
        max_vals = np.max(flat_array, axis=0)  # Max of x and y
        
        # Normalize the array
        # This is a simple min max norm 
        feat = (feat - min_vals) / (max_vals - min_vals)
        
        # Temporal Normalization	
        keypoints_array_squeezed = feat  # Shape (num_frames, num_keypoints, 3)
        for i in range(keypoints_array_squeezed.shape[0]):
            keypoints = keypoints_array_squeezed[i]  # Shape (num_keypoints, 3)

            # Check if there are enough keypoints to access the 6th one
            if keypoints.shape[0] > 6:
                # Extract x, y, and z coordinates
                x, y, z = keypoints[:, 0], keypoints[:, 1], keypoints[:, 2]

                # Subtract the 6th keypoint's x, y, and z coordinates to normalize all dimensions
                x_shifted = x - x[6]
                y_shifted = y - y[6]
                z_shifted = z - z[6]

                # Update the keypoints array with the shifted x, y, and z coordinates
                keypoints_array_squeezed[i, :, 0] = x_shifted
                keypoints_array_squeezed[i, :, 1] = y_shifted
                keypoints_array_squeezed[i, :, 2] = z_shifted

        # Reshape back to the original format (num_frames, 1, num_keypoints, 3)
        feat = keypoints_array_squeezed
        feat =  keypoints_array_squeezed[:, np.newaxis, :, :]

        return feat, conf

class HamerProcessor:
    def __init__(self):
        pass

    def temporal_segmentation(self, hand, start_frame, end_frame):    
        return hand[start_frame:end_frame]
    
    def clean_hamer_list(self, hand):
        ''' As the hamer data contains empty frames, this function filters out the empty frames and returns the hand data in a cleaned format.
        This format has a regular shape and can be converted to a numpy array
        '''
        
        filtered_hand_data = []
        json = True
        if json:
            for i in hand:
                if len(i) != 0:
                    filtered_hand_data.append(i)
        else:
            for i in hand:
                if len(i) != 0:
                    for j in i:
                        filtered_hand_data.append(j)
        
        print(filtered_hand_data)
        filtered_hand_data = np.array(filtered_hand_data)
        
        return filtered_hand_data

    
    
    def get_cleaned_hand(self, hands, handedness = 0):
        try:
            if handedness == 'L':
                hand = hands['l_hand']
            elif handedness == 'R':
                hand = hands['r_hand']
            else:
                print(handedness)
                print('Unknown handedness, check handedness list')
                exit()
        except KeyError:
            return None
        try:
            processed_hand = np.array(hand)
            
        except:
            processed_hand = self.clean_hamer_list(hand)
            
            if processed_hand.shape == (0,) or processed_hand.ndim == 2:
                print(handedness, processed_hand.shape) 
                return None
                #print('Hand data is empty')
    
        if processed_hand.shape == (0,):
                print(handedness, processed_hand.shape) 
                return None
    
        if processed_hand.ndim == 2:
            print(handedness, processed_hand.shape)
            return None
        if processed_hand.ndim == 4:
            processed_hand = processed_hand.squeeze(1)
        
        return processed_hand


class MediaPipeProcessor:
    def __init__(self, feat, conf):
        self.feat, self.conf = feat, conf

    def get_hands(self):
        """ Extract hands from MediaPipe pose format
        """
        # Remove lower body data, not visible in SB films
        feat = self.feat[:, :, :543, :]
        conf = self.conf[:, :, :543]
        
        # Only one signer in SB videos
        feat_r = feat[:, :, -21:, :].squeeze(1)  # Right hand keypoints
        feat_l = feat[:, :, -42:-21, :].squeeze(1)  # Left hand keypoints

        conf_r = conf[:, :, -21:]  # Right hand confidence
        conf_l = conf[:, :, -42:-21]  # Left hand confidence

        return feat_r, feat_l, conf_r, conf_l
    
    def crop_frames(self, data, fraction=3):
        # Calculate the number of frames
        N_frames = data.shape[0]
        
        # Calculate indices for cropping
        start_idx = N_frames // fraction  # Start after the first 1/3
        end_idx = 2 * N_frames // fraction  # End before the last 1/3
        
        # Crop the array
        cropped_data = data[start_idx:end_idx, :, :]
        
        return cropped_data

class TxtProcessor:
    def __init__(self, path):
        self.path = path

    def read_txt(self):
        with open(self.path, 'r') as f:
            data = f.readlines()
        return data

    def write_txt(self, data):
        with open(self.path, 'w') as f:
            for line in data:
                f.write(line)

    def get_2a_dict(self):
        data = self.read_txt()
        # Initialize an empty dictionary
        result_dict = {}

        # Assuming `data` is a list of strings, process each line
        for line in data:
            # Split each line by comma
            parts = line.split(',')

            # The leftmost entry is the key (strip whitespace if needed)
            key = parts[0].strip()

            # The rightmost entry is the value (strip whitespace if needed)
            value = parts[-1].strip()

            # Add the key-value pair to the dictionary
            result_dict[key] = value

        return result_dict
    
class PklProcessor:
    def __init__(self, path):
        self.path = path

    def process_directory(self):
        for filename in tqdm(files, desc="Processing files"):
            if filename.endswith("."+ pose_type):
                
                input_path = os.path.join(input_folder, filename)

        
                pose_loader = PoseFormatParser(input_path)
                pose, conf = pose_loader.read_pose()
                pose = self.pose_selector(pose.squeeze(1))
                conf = self.pose_selector(conf.squeeze(1))
                
                base, ext = os.path.splitext(filename)
                base = base.replace(".", "-")
                output_file = base + ".pkl"
                output_file = os.path.join(output_folder, output_file)            
                pkl_parser = PklParser(output_path = output_file)
                pkl_parser.pose_conf_to_pkl(pose, conf)
    

if __name__ == "__main__":
    processor = TxtProcessor('PoseTools/results/2a_handedness.txt')
    data = processor.updatde_2a()
    print(data)