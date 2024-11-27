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

from PoseTools.src.models.gca.preprocessing.hamer_utils import ManoForwardKinematics, Joint
from PoseTools.data.parsers_and_processors.datamanagment import FileConverters
fc = FileConverters()



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

def populate_empty(data):
    for key, value in data.items():
        if not value:  # Check if the list is empty
            data[key] = [np.zeros((21, 3)).tolist()]  # Add a (23, 3) zero array as a nested list
            #print(f"{key} was empty and has been populated.")
        else:
            continue #print(f"{key} is not empty.")

def process_video(video_path, URL, crop):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    
    #get fps from the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total Frames: {total_frames}, FPS: {fps}")
    crop = False
    if crop:
        start_frame = total_frames // 3  # End of the first 1/3
        end_frame = 2 * total_frames // 3  # Start of the last 1/3
    else:
        start_frame = 40
        end_frame = 60 #total_frames
    print(f"Processing frames from {start_frame} to {end_frame}")
    # Set the frame rate to 25fps
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    json_list = []
    frame_count = 0
    processed_frame_count = 0
    
    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
    #while True:
    #while frame_count < 5:
        ret, frame = cap.read()
        if not ret:
            break
        
        #print(ret, frame.shape)
        #print(frame_interval)
        # Process every nth frame to reduce framerate to 25fps
        #if processed_frame_count % frame_interval != 0:
        #    processed_frame_count += 1
        #    continue

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
                populate_empty(json_data)
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


def main_hamer(input_folder, crop = False):
    # Extract frames, process each frame, and collect the JSON data
    
    # Constants
    URL = "http://localhost:8000/process_frame/"
    PKL_OUTPUT_PATH = 'output.pkl'  # Path to store the final pkl file
    EXTERNAL_DICT_FILE = None  # Set this if needed, otherwise set to None
    POSE_TYPE = 'json'

    external_dict_file = None #'PoseTools/data/metadata/metadata_1_2s.json'
    dict_file = None #'PoseTools/data/metadata/output_2a.txt'#'PoseTools/results/handedness.txt'
    
    errors = 0
    files_to_process = [
        filename for filename in os.listdir(input_folder+'/video_files')
        if filename.endswith(".mp4") and not os.path.exists(os.path.join(input_folder+'/hamer_files', filename[:-4] + '.hamer'))
        ]
    
    for filename in files_to_process:
        video_path = os.path.join(input_folder +'/video_files', filename)
        output_folder = os.path.join(input_folder, 'hamer_files')
        output_video_path = os.path.join(output_folder, filename[:-4] + '.hamer')
        
        if os.path.exists(output_video_path):
            print(f"Skipping {filename} - already processed.")
            continue
        print(f"Processing video: {video_path}")
        try:    
            json_data = process_video(video_path, URL, crop)
        except ValueError:
            errors += 1
            continue
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
            hamerfile = filename[:-4] + ".hamer"
            #join hamerfile to hamer temp folder
            hamerfile = os.path.join(input_folder +'/hamer_files', hamerfile)
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
                    hand_data, hand_label, filename[:-4] + ".hamer", input_folder, create_gif=False
                )
            except Exception as e:
                print(f"Error processing {hand_label} in file {hamerfile}: {e}")
                continue  # Skip processing this hand
            if normalized_hand_data is not None:
                output_data[hand_label] = normalized_hand_data
                

        # Save the output data to a JSON file
        output_filename = f"normalized_"+filename[:-3]+"hamer"
        output_file = os.path.join(output_folder, output_filename)
        
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f)
                print(f"Saved normalized data to {output_file}")
        except Exception as e:
                print(f"Error saving normalized data for file {output_file}: {e}")
    
    print(f"Errors encountered: {errors}")
    
    fc = FileConverters()

    fc.to_pkl(input_folder + '/hamer_files', input_folder+'/hamer_pkl', dict_file, external_dict_file, pose_type = 'hamer')

if __name__ == "__main__":
    main()
    
