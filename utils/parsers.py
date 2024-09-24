from pose_format import Pose
import numpy.ma as ma
import numpy as np
import pickle


class PoseFormatParser:
    def __init__(self, path="A.pose"):
        self.pose_path = path

    def read_pose(self):
        """
        Load pose data from the file.

        :return: Tuple of numpy data and confidence measure.
        """
        with open(self.pose_path, "rb") as file:
            data_buffer = file.read()
        
        self.pose = Pose.read(data_buffer)
        
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

        with open(save_path, "wb") as f:
            self.pose.write(f)
        
        print(f"Pose data updated and saved to {save_path}")


class PklParser: 
    def __init__(self, path="A.pose"):
        self.pose_path = path 

    def read_pkl(self, format = 'normal'):
        with open(self.pose_path, 'rb') as f:
            data = pickle.load(f)['keypoints']
            try:
                conf = pickle.load(f)['confidence']
                if format == 'normal':
                    return data, conf
                if format == 'to_pose':
                    return np.expand_dims(data, axis=1), np.expand_dims(data, axis=1)

            except:
                if format == 'normal':
                    return data, np.ones(data.shape)
                if format == 'to_pose':
                    return np.expand_dims(data, axis=1), np.ones((data.shape[0], 1, data.shape[1]))

    
class HamerParser:
    def __init__(self, path="A.hamer"):
        self.hamer_path = path

    def read_hamer(self):
        import json

        # Load the .hamer file
        with open(self.hamer_path, 'r') as f:
            data = json.load(f)

        # Accessing the data
        l_hand = data.get('l_hand', [])
        r_hand = data.get('r_hand', [])

        hands = self.filter_hand_data([l_hand, r_hand])
        l_hand, r_hand = hands[0], hands[1]

        return l_hand, r_hand

                    
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
