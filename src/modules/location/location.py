import numpy as np
import os
from pose_format import Pose
from PoseTools.src.utils.preprocessing import PoseSelect, PoseNormalize
import pickle
from PoseTools.src.utils.plotting.hamer import plot_hamer_hand_3d
from PoseTools.src.modules.base.base import DataModule

class Location():
    def __init__(self, data):
        self.data = data 
        self.pose = data.pose
        self.com_hand()
        
    def com_hand(self):
        """
        Calculate the center of mass for a handshape.
        """
        right_hand_base_index = [16, 18, 20, 22]
        left_hand_base_index = [15, 17, 19, 21]
        self.left_hand_com = []
        self.right_hand_com = []
        for frame in self.pose:
            left_hand = frame[left_hand_base_index, :]
            self.left_hand_com.append(np.mean(left_hand, axis=0))
            right_hand = frame[right_hand_base_index, :]
            self.right_hand_com.append(np.mean(right_hand, axis=0))
        

    def determine_zone(self, hand):
        """
        Determine the zone for a single wrist based on its position.
        """
        zone = ''
        x, y, z = hand
        
        if y > 1:
            zone += ' Head -'
        elif y < 0:
            zone += ' Pelvis -'
        elif 0 <= y <= 1:
            zone += ' Neutral -'
        if x < 0:
            zone += ' Left -'
        elif x > 1:
            zone += ' Right -'
        elif 0 <= x <= 1:
            zone += ' Center -'
        return zone[:-2]
    
    def determine_location(self):
        """
        Determine the location zones for both wrists across all frames.
        Returns a list of dictionaries with zone labels for left and right wrists.
        """
        zones = []
        for idx in range(len(self.pose)):
            left_zone = self.determine_zone(self.left_hand_com[idx])
            right_zone = self.determine_zone(self.right_hand_com[idx])
            
            zones.append({
                'frame': idx,
                'left_wrist_zone': left_zone,
                'right_wrist_zone': right_zone
            })
        return zones

def main_location(data, print_results=False):
    analyzer = Location(data)
    locations = analyzer.determine_location()
    
    if print_results:
        for loc in locations[::3]:
            print(f"Frame {loc['frame']//3}: Left Wrist - {loc['left_wrist_zone']}, Right Wrist - {loc['right_wrist_zone']}")
    
    # Optionally, you can return the locations for further processing
    left = [loc['left_wrist_zone'] for loc in locations]
    right = [loc['right_wrist_zone'] for loc in locations]
    return left, right

