
import numpy as np
from tqdm import tqdm
import os
from PoseTools.data.parsers_and_processors.parsers import PoseFormatParser, TxtParser
from PoseTools.data.parsers_and_processors.processors import MediaPipeProcessor
from PoseTools.src.modules.handedness.utils.utils import calculate_center_of_mass, calculate_velocity, get_masked_arr, get_normalized_coord, extract_names_from_filtered_file
from PoseTools.src.modules.handedness.utils.graphics import plot_position, plot_velocity, plot_integrated_velocities
import pandas as pd



def process_pose_file(pose_path, process_single_file = False):
    pose_loader = PoseFormatParser(pose_path)
    pose, conf = pose_loader.read_pose()
        
    # Get the right and left hand poses

    mp_processor = MediaPipeProcessor(pose, conf)
    pose_r, pose_l, conf_r, conf_l = mp_processor.get_hands()

    # Get masked arrays
    pose_r, conf_r = get_masked_arr(pose_r, conf_r)
    pose_l, conf_l = get_masked_arr(pose_l, conf_l)

    
    # Calculate center of mass for both hands
    com_r = calculate_center_of_mass(pose_r)
    com_r_y = get_normalized_coord(com_r)

    com_l = calculate_center_of_mass(pose_l)
    com_l_y = get_normalized_coord(com_l)

    # Integrate velocity
    integrated_r = sum(com_r_y)
    integrated_l = sum(com_l_y)


    if process_single_file:
        # Plot the y-cppdomate
        plot_position(com_r_y.tolist(), com_l_y.tolist(), pose_filename)
        
                # Calculate velocity profiles
        velocity_r = calculate_velocity(com_r)
        velocity_l = calculate_velocity(com_l)

        # Plot the velocity profiles    
        plot_velocity(velocity_r, velocity_l, pose_filename)
    
    
    
    return integrated_r, integrated_l


def detect_LR(filtered_pose_files, output_path):
    integrated_velocities = []
    handedness_records = []  # To store handedness results

    #frames = get_frames("../pose/segmentation/timestamps.txt")
    
    # Open the handedness.txt file for writing
    with open(output_path, "w") as handedness_file:
        i = 0
        l_hand = 0 
        
        # Process only the filtered pose files
        for pose_file in tqdm(filtered_pose_files, desc="Processing handedness 1 pose files"):
            pose_path = os.path.join(directory_path, pose_file)
            
            #start, stop = frames[pose_file[:-5]]

            
            
            integrated_r, integrated_l = process_pose_file(pose_path)
            
            # Determine handedness and write to file
            if integrated_l > integrated_r:
                handedness_records.append(f"{pose_file}, L\n")
                handedness_file.write(f"{pose_file[:-5]}, L\n")
                l_hand += 1
            else:
                handedness_records.append(f"{pose_file}, R\n")
                handedness_file.write(f"{pose_file[:-5]}, R\n")
            
            integrated_velocities.append((integrated_r, integrated_l))
            
            i += 1
        
        print('completed processing all files')
        
        print('Percentage of signs executed with the left hand: ', np.round(100*l_hand / i,2), '%')

def get_glosses_with_handedness_1(data):
    """
    Extracts the 'Annotation ID Gloss Dutch' for entries with 'Handedness' equal to 1.
    
    Parameters:
    - data: The JSON data as a list of entries (with nested structures).
    
    Returns:
    - A list of 'Annotation ID Gloss Dutch' where 'Handedness' is 1.
    """
    glosses = []
    
    for entry in data:
        # Each entry is a dictionary with a single key-value pair
        # The key is the unique ID (e.g., '49307'), and the value is the actual data dictionary
        for entry_id, entry_data in entry.items():
            # Check if 'Handedness' is present and equals '1'
            if 'Handedness' in entry_data and entry_data['Handedness'] == '1':
                gloss = entry_data.get('Annotation ID Gloss: Dutch', None)
                if gloss:  # Ensure the gloss exists
                    glosses.append(gloss)
    
    return glosses

def process_directory(directory_path, output_file, filtered_file_path, method = 'detect_LR'):

    # Get a list of all .pose files in the directory that match the filtered names
    pose_files = [f[:-5] for f in os.listdir(directory_path) if f.endswith('.pose')]
    
    if method == 'detect_LR':
        parser = TxtParser('PoseTools/data/metadata/glosses_meta.json')
        data = parser.read_json()

        handedness_1 = get_glosses_with_handedness_1(data)
        print('number of glosses with handedness 1: ', len(handedness_1))
        
        filtered_pose_files = [f + '.pose' for f in pose_files if f in handedness_1]
        print('Number of files to process: ', len(filtered_pose_files))

        integrated_velocities = detect_LR(filtered_pose_files, filtered_file_path)
        
        # Plot integrated velocities for all files
        plot_integrated_velocities(integrated_velocities, output_file)

def process_directory_h1(df, pose_dir, output_dir):
    # List all .pkl files in the pose_dir
    filenames = set([file[:-6] for file in os.listdir(pose_dir) if file.endswith('.pkl')])
    for filename in filenames:
        # Construct full file path
        filepath = os.path.join(pose_dir, filename)
        filepath_L = filepath + '-L.pkl'
        filepath_R = filepath +'-R.pkl' 
        
        # Load the .pkl file (assuming it's a pandas DataFrame)
        with open(filepath_L, 'rb') as f:
            hand_L = pd.read_pickle(f)['keypoints']
        with open(filepath_L, 'rb') as f:
            hand_R = pd.read_pickle(f)['keypoints']
        
        # Call detect_LR (make sure filtered_pose_files and filtered_file_path are defined)
        integrated_velocities = detect_LR(hand_L, hand_R, file_path)
        
        # Save the result to the output directory
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'wb') as f_out:
            pd.to_pickle(integrated_velocities, f_out)


if __name__ == "__main__":
    # Load pose data
    import os 
    directory_path = "../signbank_videos/"  # Path to the directory containing .pose files
    output_file = "PoseTools/src/modules/handedness/graphics/integrated_velocity_barplot.png"  # Output file for the bar plot
    filtered_file_path = "PoseTools/src/modules/handedness/model/h1s.txt"  # Metadata file for the dataset

    #pose_filename = "DEN-HAAG-B"
    #pose_path = os.path.join(directory_path, pose_filename + ".pose")
    #process_pose_file(pose_path, process_single_file = True)
    
    process_directory(directory_path, output_file, filtered_file_path)