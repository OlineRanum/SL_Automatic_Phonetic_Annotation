import os
import json
from tqdm import tqdm
from PoseTools.utils.processors import HamerProcessor
from PoseTools.utils.parsers import PoseFormatParser, PklParser, HamerParser

class FileOrganizer:
    def __init__(self):
        pass

    # Function to check if a corresponding .pkl file exists
    def check_pkl_exists(self, directory, video_id):
        pkl_filename = f"{video_id}.pkl"
        return os.path.exists(os.path.join(directory, pkl_filename))

    # Function to process the JSON file and check for missing .pkl files
    def process_file(self, file_path, directory):
        # Load the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        missing_glosses = []
        
        counter = 0
        # Iterate through the glosses and their instances
        for gloss_data in data:
            gloss = gloss_data.get("gloss", "")
            for instance in gloss_data.get("instances", []):
                video_id = instance.get("video_id", "")
                if video_id:
                    pkl_exists = self.check_pkl_exists(directory, video_id)
                    if not pkl_exists:
                        counter += 1
                        missing_glosses.append({
                            "gloss": gloss,
                            "video_id": video_id
                        })
        
        
        return missing_glosses, counter

    '''
    # Example usage
    json_file_path = 'filtered_metadata_reduced.json'
    pkl_directory = '../hamer_pkl'

    # Process the file and get missing glosses
    missing_glosses, counter = process_file(json_file_path, pkl_directory)

    # Output missing glosses
    for gloss_info in missing_glosses:
        print(f"Gloss: {gloss_info['gloss']}, Video ID: {gloss_info['video_id']} is missing .pkl file.")

    print('Total missing glosses:', counter)
    '''


class FileConverters:
    def __init__(self):
        pass

    def to_pkl(self, input_folder, output_folder, dict_file, external_dict_file, pose_type = 'pose', multi_hands = False, convert2a = False):
        """
        Iterate over all pose files in the input folder, preprocess them, and save them to the output folder.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        files = os.listdir(input_folder)
        if pose_type == 'pose_format':
            pkl_parser = PklParser(output_folder)
            for filename in tqdm(files, desc="Processing files"):
                if filename.endswith("."+ pose_type):
                    
                    input_path = os.path.join(input_folder, filename)

                    #print(f"Processing {filename}...")
                    
                    # Save processed pose to output folder
                    if pose_type == 'pose_format':
                        pose_loader = PoseFormatParser(input_path)
                        pose, conf = pose_loader.read_pose()
                        pkl_parser.pose_conf_to_pkl(pose, conf)
            
        if pose_type == 'hamer' or pose_type == 'json':
            hamer_parser = HamerParser(input_folder, output_folder)
            if multi_hands:

                hamer_parser.hamer_to_pkl(pose_type, dict_file, external_dict_file, multi_handedness_classes = True)
            elif convert2a:
                hamer_parser.hamer_to_pkl_2a(pose_type, dict_file, external_dict_file, convert2a = convert2a)
            else:
                hamer_parser.hamer_to_pkl(pose_type, dict_file)                


        
    def to_pose(self, input_folder, output_folder, pose_type = 'pkl', multi_hands = False):
        """
        Iterate over all pose files in the input folder, preprocess them, and save them to the output folder.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        files = os.listdir(input_folder)
        if pose_type == 'pkl':
            pkl_parser = PklParser(output_folder)
            for filename in tqdm(files, desc="Processing files"):
                if filename.endswith("."+ pose_type):
                    
                    input_path = os.path.join(input_folder, filename)
                    pose, conf = pkl_parser.read_pkl(format = 'to_pose', input_path = input_path)
                    
                    print(pose.shape, conf.shape)
                    # Save processed pose to output folder
                    print(output_folder)
                    pose_folder = '../signbank_videos'
                    filename_ = filename.replace(".pkl", "")[:-2] + ".pose"
                    pose_path = os.path.join(pose_folder, filename_)
                    
                    
                    pose_loader = PoseFormatParser(pose_path)
                    pose, conf = pose_loader.read_pose()
                    output_path = os.path.join(output_folder, filename.replace(".pkl", ".pose"))
                    pose_loader.write_pose(pose, conf, save_path = output_path)
                    