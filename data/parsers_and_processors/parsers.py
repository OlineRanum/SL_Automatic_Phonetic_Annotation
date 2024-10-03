from pose_format import Pose
import numpy.ma as ma
import numpy as np
import pickle, os, json
from PoseTools.data.parsers_and_processors.processors import HamerProcessor
from tqdm import tqdm

class PoseFormatParser:
    def __init__(self, path="A.pose"):
        self.pose_path = path

    def read_pose(self, n_points = None):
        """
        Load pose data from the file.

        :return: Tuple of numpy data and confidence measure.
        """
        try:
            # Attempt to open the initial path
            with open(self.pose_path, "rb") as file:
                data_buffer = file.read()
        except FileNotFoundError:
            import re
            # Try replacing the first '-' after the number with a '.'
            new_path = re.sub(r'(\d)-', r'\1.', self.pose_path, 1)
            print(f"First fallback path: {new_path}")
            
            try:
                with open(new_path, "rb") as file:
                    data_buffer = file.read()
            except FileNotFoundError:
                # Try replacing '-PL' with '.PL'
                new_path = re.sub(r'-PL', r'.PL', new_path, 1)
                print(f"Second fallback path: {new_path}")
                
                try:
                    with open(new_path, "rb") as file:
                        data_buffer = file.read()
                except FileNotFoundError:
                    return None, None

        self.pose = Pose.read(data_buffer, n_points = n_points)

        data = self.pose.body.data.data
        conf = self.pose.body.confidence

        return data, conf


    def write_pose(self, data, conf, save_path="updated_pose.pose"):
        """
        Save updated pose data to a file.
        """

        if isinstance(data, np.ndarray):  
            mask = conf == 0  # 0 means no-confidence (masked)
            stacked_mask = np.stack([mask] * data.shape[-1], axis=3)
            masked_data = ma.masked_array(data, mask=stacked_mask)

        else:
            masked_data = data  

        self.pose.body.data = masked_data
        self.pose.body.confidence = conf
        self.pose.body.frames = data.shape[0]
        

        with open(save_path, "wb") as f:
            self.pose.write(f)
        
        #print(f"Pose data updated and saved to {save_path}")


class PklParser: 
    def __init__(self, input_path="A.pose", output_path = "A.pose"):
        self.input_path = input_path
        self.output_path = output_path

    def read_pkl_simple(self):

        # Open and load the pickle file
        with open(self.input_path, 'rb') as file:
            data = pickle.load(file)

        # Now 'data' contains the contents of your pkl file
        print('Pickled Data---------------')
        print(data)
        print(type(data['keypoints']))
        print(type(data['confidences']))
        print(data['keypoints'].shape)
        print(data['confidences'].shape)
        return data

    def read_pkl(self, format = 'normal', input_path = None):
        if input_path is not None:
            self.input_path = input_path
        
        with open(self.input_path, 'rb') as f:
            # Load the whole data dictionary in one go
            data_dict = pickle.load(f)
            data = data_dict.get('keypoints', None)
            conf = data_dict.get('confidence', None)
            
            if data is None:
                raise ValueError("Key 'keypoints' not found in the pickle file")
            
            # If confidence data is not available, use default ones
            if conf is None:
                conf = np.ones((data.shape[0], data.shape[1]))

            if format == 'normal':
                return data, conf
            elif format == 'to_pose':
                return np.expand_dims(data, axis=1), np.expand_dims(conf, axis=1)
            else:
                raise ValueError("Unknown format specified")
                    
    def pose_conf_to_pkl(self, data, conf):
        """
        Save updated pose data to a file.
        """

        if isinstance(data, np.ndarray) and isinstance(conf, np.ndarray):
            # Create the dictionary to be saved
            data_dict = {
                'keypoints': data,
                'confidences': conf
            }

            # Save the dictionary to a pickle file
            with open(self.output_path, 'wb') as file:
                pickle.dump(data_dict, file)
            
            #print(f"Data successfully saved to {self.output_path}")
        else:
            raise TypeError("Both data and conf must be numpy ndarrays.")


    
class HamerParser:
    def __init__(self, input_folder = None, output_folder= None):
        self.source_dir = input_folder
        self.destination_dir = output_folder

        self.processor = HamerProcessor()
        self.corrupted_files = 0

    def read_hamer(self):
        import json

        # Load the .hamer file
        with open(self.hamer_path, 'r') as f:
            data = json.load(f)

        # Accessing the data
        l_hand = data.get('l_hand', [])
        r_hand = data.get('r_hand', [])

        hands = self.processor.filter_hand_data([l_hand, r_hand])
        l_hand, r_hand = hands[0], hands[1]

        return l_hand, r_hand
    

    def update_handedness_dict_from_json(self, json_file, handedness_dict):
        # Load the JSON data
        with open(json_file, 'r') as f:
            gloss_data = json.load(f)

        # Iterate over gloss entries
        for gloss_entry in gloss_data:
            gloss = gloss_entry['gloss']
            instances = gloss_entry['instances']

            # Iterate over instances for the current gloss
            for instance in instances:
                video_id = instance['video_id']

                # Check if the video_id exists in the handedness_dict
                if video_id not in handedness_dict:
                    # Extract the handedness from the video_id by getting the last character (either 'L' or 'R')
                    handedness = video_id.split('-')[-1]

                    # Add new entry to the handedness_dict with video_id as the key and handedness ('L' or 'R') as the value
                    handedness_dict[video_id] = handedness

        return handedness_dict

    def extend_handedness_dict(self, handedness_dict, dict_file):
        extended_dict = {}

        # Loop through the existing handedness dictionary
        for video_id, handedness in handedness_dict.items():
            # Get the corresponding gloss from gloss_dict
            
            if video_id:  # If the gloss exists for this video_id
                # Modify the key to be of the form gloss-handedness
                new_key = f"{video_id}-{handedness}"
                extended_dict[new_key] = handedness
        
        json_file = dict_file
        extended_dict = self.update_handedness_dict_from_json(json_file, extended_dict)
        
        return extended_dict

    def process_hamer_file(self, filepath, filename, handedness):
        # Load the .hamer JSON file
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except:
            try:
                with open(filepath.replace('-','.'), 'r') as f:
                    data = json.load(f)
            except:
                try:
                    # Replace the last '.' with '-' (handle cases like 6.ORD-B)
                    filepath_parts = filepath.rpartition('.')
                    modified_filepath = filepath_parts[0] + '-' + filepath_parts[2]
                    print(modified_filepath)
                    with open(modified_filepath, 'r') as f:
                        data = json.load(f)
                except FileNotFoundError:
                    print(f"File not found in any format: {filepath}")



        data_filtered = self.processor.get_cleaned_hand(data, handedness)
        if data_filtered is None:
            print('File Corrupted ', filename)
            self.corrupted_files += 1
            return None
            
        
        # Create a dictionary with the 'keypoints' key containing the l_hand array
        hand_data = {
            "keypoints": data_filtered
            
        }

        # Create the output filepath for the .pkl file
        output_filename = os.path.splitext(filename)[0] 
        if "normalized_" in output_filename:
            output_filename = output_filename.split('_', 1)[1]
        if "_segment" in filename:
            output_filename = output_filename.split('_', 1)[0]
        if '.' in filename:
            output_filename = output_filename.replace('.', '-')
        output_filename = output_filename + f"-{handedness}"
        if ".pkl" not in output_filename:
            output_filename = output_filename + ".pkl"
        
        output_filepath = os.path.join(self.destination_dir, output_filename)

        # Save the dictionary as a .pkl file
        with open(output_filepath, 'wb') as pkl_file:
            pickle.dump(hand_data, pkl_file)
    
        
    
    def hamer_to_pkl(self, pose_type, dict_file, external_dict_file = None, multi_handedness_classes = False):
        """
        Convert .hamer JSON files in source_dir to .pkl format with 'keypoints' key containing 'l_hand' data 
        and save them in destination_dir.
        
        Args:
            source_dir (str): Path to the directory containing the .hamer JSON files.
            destination_dir (str): Path to the directory where .pkl files will be saved.
        """
        # Ensure the destination directory exists
        os.makedirs(self.destination_dir, exist_ok=True)

        handedness_dict = TxtParsers(dict_file).get_handedness_dict()
        
        if multi_handedness_classes:
            handedness_dict = self.extend_handedness_dict(handedness_dict, external_dict_file)

        total = 1
        left = 0
        # Loop through all .hamer files in the source directory
        files = os.listdir(self.source_dir)
        for filename in tqdm(files, desc="Converting files"):
            if filename.endswith("." + pose_type):
                filepath = os.path.join(self.source_dir, filename)
                
                gloss = filename[:-(len(pose_type) + 1)]
                
                if "normalized_" in gloss:
                    gloss = gloss.split('_', 1)[1]
                if "_segment" in gloss:
                    gloss = gloss.split('_', 1)[0]
                
                if not multi_handedness_classes:
                    handedness = handedness_dict.get(gloss)

                    if handedness is None:
                        #print(f"Unknown handedness for gloss: {gloss}")
                        continue
                    else:
                        self.process_hamer_file( filepath, filename, handedness)
                    
                    #print(f"Converted {filename} to {output_filename}")
                else:
                    
                    if handedness_dict.get(gloss + '-L') is not None:
                        
                        self.process_hamer_file(filepath, filename, 'L')
                        left += 1
                        total += 1
                    
                    if handedness_dict.get(gloss + '-R') is not None:
                        
                        self.process_hamer_file(filepath, filename, 'R')
                        total += 1
            
        print('Total number of processed files', total)
        print('Percentage of left handed signs', left/total)


    def hamer_to_pkl_2a(self, pose_type, dict_file, external_dict_file = None, convert2a = True):
        """
        Convert .hamer JSON files in source_dir to .pkl format with 'keypoints' key containing 'l_hand' data 
        and save them in destination_dir.
        
        Args:
            source_dir (str): Path to the directory containing the .hamer JSON files.
            destination_dir (str): Path to the directory where .pkl files will be saved.
        """
        # Ensure the destination directory exists
        os.makedirs(self.destination_dir, exist_ok=True)
        
        handedness_dict = TxtParsers(dict_file).get_handedness_dict(convert2a)
        total = 1
        
        for filename in tqdm(handedness_dict, desc="Converting files"):
            filepath = os.path.join(self.source_dir, 'normalized_' + filename + "_segment." + pose_type)
            
            
            for handedness in ['L', 'R']:
                self.process_hamer_file( filepath, filename, handedness)
                
            total += 1       
        print('Number of corrupted files', self.corrupted_files)    
        print('Total number of processed files', total)
        




                    
class EafParser:
    def __init__(self, path="A.eaf"):
        self.eaf_file = path

    def parse_eaf(self):
        import xml.etree.ElementTree as ET
        # Parse the XML file
        tree = ET.parse(self.eaf_file)
        root = tree.getroot()

        # Dictionary to store time slots
        time_slots = {}

        # Extract time slots and store them in the dictionary
        for time_slot in root.findall(".//TIME_SLOT"):
            time_id = time_slot.get("TIME_SLOT_ID")
            time_value = int(time_slot.get("TIME_VALUE"))
            time_slots[time_id] = time_value
        print(time_slots)
        

       # Find the annotations in the SENTENCE tier
        sentence_tier = root.find(".//TIER[@TIER_ID='SENTENCE']")

        # Collect the start and end times of the sentence annotations
        sentence_times = []
        if sentence_tier is not None:
            for annotation in sentence_tier.findall(".//ALIGNABLE_ANNOTATION"):
                time_slot_ref1 = annotation.get("TIME_SLOT_REF1")
                time_slot_ref2 = annotation.get("TIME_SLOT_REF2")
                start_time_ms = time_slots.get(time_slot_ref1)
                end_time_ms = time_slots.get(time_slot_ref2)
                sentence_times.append((start_time_ms, end_time_ms))

                print(f"Sentence annotation (ID: {annotation.get('ANNOTATION_ID')}):")
                print(f"Start ref: {time_slot_ref1}")
                print(f"End ref: {time_slot_ref2}")
                print(f"Start time: {start_time_ms} ms")
                print(f"End time: {end_time_ms} ms\n")

        return start_time_ms, end_time_ms

    def get_frames(self, file_path):
        """ Read frames from a txt file
        """
        gloss_dict = {}
        
        # Open and read the file
        with open(file_path, 'r') as file:
            file.readline()  # Skip the header
            for line in file:
                # Split the line by commas
                gloss, start_frame, end_frame = line.strip().split(',')
                
                # Convert start_frame and end_frame to integers
                gloss_dict[gloss] = [int(start_frame), int(end_frame)]
        
        return gloss_dict


class TxtParser:
    def __init__(self, file_path="A.hamer"):
        self.txt_file_path = file_path

    def get_handedness_dict(self, convert2a = False):
        if convert2a:
            result_dict = []
        else:
            result_dict = {}
        
        with open(self.txt_file_path, 'r') as file:
            for line in file:
                line = line.strip()
                
                # Split the line at the comma
                if ',' in line:
                    key, value = line.split(',', 1)
                    
                    # Strip any extra whitespace around the key and value
                    key = key.strip()
                    value = value.strip()
                    
                    # Add key-value pair to the dictionary
                    if convert2a:
                        result_dict.append(key)
                    else:
                        result_dict[key] = value

        return result_dict
    
    def read_json(self):
        with open(self.txt_file_path, 'r') as f:
            data = json.load(f)
        return data


