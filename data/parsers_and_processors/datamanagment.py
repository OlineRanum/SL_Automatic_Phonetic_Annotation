import os
import json
from tqdm import tqdm
from PoseTools.data.parsers_and_processors.processors import PklProcessor
from PoseTools.data.parsers_and_processors.parsers import PoseFormatParser, PklParser, HamerParser
from PoseTools.src.utils.preprocessing import PoseSelect

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


class FileConverters:
    def __init__(self):
        self.pose_selector = PoseSelect(preset="mediapipe_holistic_minimal_27")

    def preprocess_pose(self, pose):
        pose = self.pose_selector.clean_keypoints(pose)
        pose = self.pose_selector.get_keypoints_pose_and_hands(pose)
        return self.pose_selector(pose)


    def to_pkl(self, input_folder, output_folder, dict_file = None, external_dict_file= None, pose_type = 'pose', multi_hands = False, convert2a = False):
        """
        Iterate over all pose files in the input folder, preprocess them, and save them to the output folder.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
                    
        files = os.listdir(input_folder)
        if pose_type == 'pose':
            pkl_parser = PklProcessor(output_folder)
            #pkl_parser.process_directory(input_folder, output_folder, files)

            for filename in tqdm(files, desc="Processing files"):
                if filename.endswith("."+ pose_type):
                    
                    input_path = os.path.join(input_folder, filename)

            
                    pose_loader = PoseFormatParser(input_path)
                    pose, conf = pose_loader.read_pose()
                    
                    pose = self.preprocess_pose(pose)
                    conf = self.preprocess_pose(conf)
                    print(pose.shape, conf.shape)
                    
                    base, ext = os.path.splitext(filename)
                    base = base.replace(".", "-")
                    output_file = base + ".pkl"
                    output_file = os.path.join(output_folder, output_file)            
                    pkl_parser = PklParser(output_path = output_file)
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
                    output_path = os.path.join(output_folder, filename.replace(".pkl", ".pose"))

                    if os.path.exists(output_path):
                        continue
                    
                    pose, conf = pkl_parser.read_pkl(format = 'to_pose', input_path = input_path)
                    if pose is not None:
                        #pose, conf = self.pose_selector(pose), self.pose_selector(pose)
                        
                        filename_ = filename.replace(".pkl", "") + ".pose"

                    
                        pose_folder = '../signbank_videos'
                        
                        
                        pose_path = os.path.join(pose_folder, filename_)
                                 
                        pose_loader = PoseFormatParser(pose_path)
                        
                        _, _ = pose_loader.read_pose()
                        
                        pose_loader.write_pose(pose, conf, save_path = output_path)
                        
                        