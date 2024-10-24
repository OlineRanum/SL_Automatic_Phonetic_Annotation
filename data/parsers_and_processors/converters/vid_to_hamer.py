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

from PoseTools.src.models.graphTransformer.preprocessing.hamer_utils import ManoForwardKinematics, Joint
from PoseTools.data.parsers_and_processors.datamanagment import FileConverters
fc = FileConverters()

# Constants
URL = "http://localhost:8000/process_frame/"
PKL_OUTPUT_PATH = 'output.pkl'  # Path to store the final pkl file
EXTERNAL_DICT_FILE = None  # Set this if needed, otherwise set to None
POSE_TYPE = 'json'

input_folder = "PoseTools/src/models/graphTransformer/test_data/vids/"
output_folder = input_folder
external_dict_file = None #'PoseTools/data/metadata/metadata_1_2s.json'
dict_file = None #'PoseTools/data/metadata/output_2a.txt'#'PoseTools/results/handedness.txt'

# Resize frame while keeping aspect ratio
def resize_with_aspect_ratio(frame, target_width, target_height):
    original_height, original_width = frame.shape[:2]

    # Calculate the scaling factors for width and height
    scale_width = target_width / original_width
    scale_height = target_height / original_height

    # Choose the smallest scale factor to ensure the entire image fits within target dimensions
    scale = min(scale_width, scale_height)

    # Resize the image
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_frame = cv2.resize(frame, (new_width, new_height))

    return resized_frame
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    
    #get fps from the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")
    # Set the frame rate to 25fps
    fps = 25
    frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / fps)
    
    json_list = []
    frame_count = 0
    processed_frame_count = 0
    while True:
    #while frame_count < 5:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every nth frame to reduce framerate to 25fps
        if processed_frame_count % frame_interval != 0:
            processed_frame_count += 1
            continue

        target_width = 640
        target_height = 360

        # Resize the frame while maintaining aspect ratio
        resized_frame = resize_with_aspect_ratio(frame, target_width, target_height)

        # Pad the resized frame to exactly 720p, if necessary
        delta_w = target_width - resized_frame.shape[1]
        delta_h = target_height - resized_frame.shape[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # Padding with black (0, 0, 0)
        color = [0, 0, 0]
        frame = cv2.copyMakeBorder(resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        #for debug, save frame to temp folder
        #cv2.imwrite(os.path.join(input_folder, f"frame_{frame_count}.png"), frame)

        # Encode frame to memory as PNG for sending via HTTP
        success, frame_encoded = cv2.imencode('.png', frame)
        if not success:
            print(f"Failed to encode frame {frame_count}")
            continue

        # Send the frame to the URL
        response = requests.post(
            URL,
            files={'file': ('frame.png', frame_encoded.tobytes(), 'image/png')},
            data={'video_filename': video_path}
        ) 
              
        if response.status_code == 200:
            try:
                json_data = response.json()  # Assuming the response is in JSON format
                # print(json_data)
                json_list.append(json_data)
            except json.JSONDecodeError:
                print(f"Failed to decode JSON response for frame {frame_count}")
        else:
            print(f"Failed to process frame {frame_count}, status code: {response.status_code}")

        frame_count += 1
        processed_frame_count += 1
        
        # if frame_count == 1:
        #     break

    cap.release()

    return json_list


if __name__ == "__main__":
    # Extract frames, process each frame, and collect the JSON data
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):  # Assuming your videos are in .mp4 format
            video_path = os.path.join(input_folder, filename)
            print(f"Processing video: {video_path}")
            json_data = process_video(video_path)
            concatenated_l_hand = []
            concatenated_r_hand = []

            #we have to convert the json by collecting all l_hand data from all frames and put them together in a list under l_hand, same goes for r_)hand
            for item in json_data:
                # Extract the l_hand and r_hand values and extend the concatenated lists
                concatenated_l_hand.append(item["l_hand"])
                concatenated_r_hand.append(item["r_hand"])

                # Create the new dictionary with all concatenated l_hand and r_hand data
                json_data = {"l_hand": concatenated_l_hand, "r_hand": concatenated_r_hand}
                
                mano_fk = ManoForwardKinematics()

                # Save the JSON data to a temporary file
                #name temp file with filetype hamer
                hamerfile = "temp_hamer.hamer"
                #join hamerfile to hamer temp folder
                hamerfile = os.path.join(input_folder, hamerfile)
                #write json data to temp file
                with open(hamerfile, 'w') as f:
                    json.dump(json_data, f)
                
            
            output_data = {}
            

            # Process each hand
            for hand_label in ['l_hand', 'r_hand']:
                #get hand_label from hamerfile
                hand_data = json_data[hand_label]
                
                # Process and normalize the hand data, create GIFs
                try:
                    normalized_hand_data = mano_fk.normalize_and_save(
                        hand_data, hand_label, "temp_hamer.hamer", input_folder, create_gif=False
                    )
                except Exception as e:
                    print(f"Error processing {hand_label} in file {hamerfile}: {e}")
                    continue  # Skip processing this hand
                if normalized_hand_data is not None:
                    output_data[hand_label] = normalized_hand_data
                    

            # Save the output data to a JSON file
            output_filename = f"norm/normalized_"+filename[:-3]+"hamer"
            output_file = os.path.join(input_folder, output_filename)
            
            try:
                with open(output_file, 'w') as f:
                    json.dump(output_data, f)
                    print(f"Saved normalized data to {output_file}")
            except Exception as e:
                    print(f"Error saving normalized data for file {output_file}: {e}")

    
    fc = FileConverters()

    fc.to_pkl(input_folder + 'norm', output_folder + 'norm/pkl', dict_file, external_dict_file, pose_type = 'hamer')
    
