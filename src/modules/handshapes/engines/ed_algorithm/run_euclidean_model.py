
import json, os, re
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from PoseTools.data.parsers_and_processors.parsers import PklParser, MetadataParser
from collections import Counter
from PoseTools.src.modules.handedness.utils.graphics import read_dict_from_txt, plot_multiple_hands_from_dict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

class EvaluatePoses:
    def __init__(self, df, reference_poses, pose_dir='/mnt/fishbowl/gomer/oline/hamer_cleaned'):
        self.pose_dir = pose_dir
        self.df = df
        self.reference_poses = reference_poses
        self.gloss_mapping = read_dict_from_txt('/home/gomer/oline/PoseTools/data/metadata/output/global_value_to_id.txt')
        print('these references: ', self.reference_poses.keys())
        print(self.gloss_mapping)
        handshapes = df['Handshape'].unique()
        values = []
        for handshape in handshapes:
            values.append(str(self.get_key_from_value(handshape)))
        self.reference_poses = {key: self.reference_poses[key] for key in values if key in self.reference_poses}
        print('these references: ', self.reference_poses.keys())
        

    

    def calculate_euclidean_distance(self, pose, reference_pose):
        """
        Calculates the Euclidean distance between a pose and a reference pose for each keypoint.
        
        Parameters:
        - pose: A numpy array of shape (21, 3), representing the pose for a frame.
        - reference_pose: A numpy array of shape (21, 3), representing the reference handshape pose.
        
        Returns:
        - The Euclidean distance between the pose and the reference pose.
        """
        return np.linalg.norm(pose - reference_pose, axis=1).mean()  # Average Euclidean distance for all keypoints

    def get_key_from_value(self, value):
        """
        Returns the key in the gloss_mapping dictionary that corresponds to the given value.
        
        Parameters:
        - gloss_mapping: A dictionary where keys are integers and values are handshapes.
        - value: The value (handshape) for which the key is needed.
        
        Returns:
        - The key corresponding to the given value, or None if the value is not found.
        """
        
        for key, val in self.gloss_mapping.items():
            
            if val == value:
                return key
        return None  # Return None if the value is not found
    
    def get_value_from_key(self, value):
        """
        Returns the key in the gloss_mapping dictionary that corresponds to the given value.
        
        Parameters:
        - gloss_mapping: A dictionary where keys are integers and values are handshapes.
        - value: The value (handshape) for which the key is needed.
        
        Returns:
        - The key corresponding to the given value, or None if the value is not found.
        """
        
        for key, val in self.gloss_mapping.items():
            
            if key == value:
                return val
        return None  # Return None if the value is not found
    
    def handshape_per_frame(self, pose_data):
        """
        Evaluates each frame by calculating the Euclidean distance to the reference handshapes,
        and determines the median handshape across all frames.
        """
        num_frames = pose_data.shape[0]
        closest_handshapes = []  # List to store the closest handshape per frame
        distances = [] 
        for frame_idx in range(num_frames):
            pose = pose_data[frame_idx]  # Pose at the current frame (21, 3)
            
            # Calculate the Euclidean distance to both handshapes
            distances = []
            keys = []
            for key, reference_pose in self.reference_poses.items():
                distance = self.calculate_euclidean_distance(pose, reference_pose)
                keys.append(key)
                distances.append(distance)

            closest_handshape = self.gloss_mapping[int(keys[np.argmin(np.array(distances))])]
            
            closest_handshapes.append(closest_handshape)

        
        
        handshape_counter = Counter(closest_handshapes)
        try:
            median_handshape = handshape_counter.most_common(1)[0][0]  # Get the most frequent handshape
            
        except IndexError:
            print('!!!!!!!!!')
            median_handshape = 'h1'
        return median_handshape
    


    def get_handshape(self, video_id):
        video_id = os.path.join(self.pose_dir, video_id) + '.pkl'
        
        video_id = os.path.abspath(video_id)
        if os.path.exists(video_id):
            parser = PklParser(input_path=video_id)
            pose_data, _ = parser.read_pkl()  # pose_data shape is (num_frames, 21, 3)
            return self.handshape_per_frame(pose_data)
        else: 
            return None

    def evaluate_poses(self):
        """
        Evaluates poses for each row in the DataFrame by checking if pose files exist for -L and -R,
        and counts the number of existing files.
        """

        output_file = 'PoseTools/results/euclidean_sb.txt'
        self.gloss_mapping = read_dict_from_txt('/home/gomer/oline/PoseTools/data/metadata/output/global_value_to_id.txt')
        
        truth = []
        pred = []


        print('Number of classes ', len(df['Handshape'].unique()))
        
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing Poses"):
            # Get the video_id from the row
            filename = row['filename']
            handshape = row['Handshape']    
            hand = self.get_handshape(filename)

            if hand is not None:
                truth.append(handshape)
                pred.append(hand)

        truth, pred = np.array(truth), np.array(pred)
        print(len(set(truth)))
        print(set(truth))
        
        print(len(set(pred)))
        print(set(pred))
        # Ensure the labels are consistent between truth and predictions using LabelEncoder
        label_encoder = LabelEncoder()
        
        # Fit the encoder on the combined set of unique handshapes from both truth and pred
        all_labels = np.unique(np.concatenate([truth, pred]))
        
        label_encoder.fit(all_labels)
        

        # Transform the truth and pred into numerical values
        truth_encoded = label_encoder.transform(truth)
        pred_encoded = label_encoder.transform(pred)

        print(len(set(truth)))
        print(set(truth))
        print(len(set(pred)))
        print(set(pred))
        missing_preds = set(truth) - set(pred)
        if missing_preds:
            print(f"Handshapes in truth but missing in predictions: {missing_preds}")
            exit()
        
        # Compute total accuracy
        total_correct = np.sum(truth == pred)
        total_predictions = len(truth)
        accuracy = total_correct / total_predictions
        
        print(f"Total Accuracy: {accuracy * 100:.2f}%")

        # Create the confusion matrix with the encoded labels
        cm = confusion_matrix(truth_encoded, pred_encoded)

        # Normalize the confusion matrix
        with np.errstate(invalid='ignore', divide='ignore'):
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN values with 0

        cm_percentage = (cm_normalized * 100).astype(int)

        # Plotting the confusion matrix
        fsize = 15
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm_percentage, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,
                    annot_kws={"size": fsize})
        
        plt.xticks(rotation=45, ha='right', fontsize=fsize)
        plt.yticks(rotation=0, fontsize=fsize)

        plt.xlabel('Predicted Handshape', fontsize=fsize +2)
        plt.ylabel('True Handshape', fontsize=fsize + 2)
        plt.title('Normalized Confusion Matrix for Handshape Classification', fontsize=fsize+4)
        
        plt.tight_layout()
        plt.savefig('/home/gomer/oline/PoseTools/src/models/euclidean_model/confusion_matrix_sb.png')

if __name__ == "__main__":
    #df = pd.read_csv('/home/gomer/oline/PoseTools/src/models/euclidean_model/ground_truth_handshape.csv', delimiter=',', na_values='Unknown')
    # Load the object from the file
    #df = df.rename(columns={'gloss': 'video_id', 'handedness': 'Sign Type', 'strong_hand': 'Handshape', 'weak_hand': 'Nondominant Handshape'})
    with open('/home/gomer/oline/PoseTools/src/models/euclidean_model/reference_poses.pkl', 'rb') as file:
        reference_poses = pickle.load(file)
    print(reference_poses.keys())
    parser = MetadataParser('/home/gomer/oline/PoseTools/data/metadata/output/35c/35c_test.json')
    
    gloss_dict = parser.read_metadata()
    
    df = pd.DataFrame(gloss_dict, columns=['filename', 'Split', 'Source', 'Sign Type', 'Handshape'])
    #df = df[df['Sign Type'] == '1']
    print('Number of classes ', len(df['Handshape'].unique()))
    # Create an instance of the class
    evaluator = EvaluatePoses(df, reference_poses)

    # Call the evaluate_poses method to print video IDs
    evaluator.evaluate_poses()


