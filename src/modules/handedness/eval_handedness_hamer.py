import argparse
import os 
from PoseTools.src.modules.handedness.euclidean_model.euclidean_model import EvaluatePoses
import json
import pandas as pd
from PoseTools.src.modules.handedness.euclidean_model.euclidean_model import EvaluatePoses
import pickle
from PoseTools.src.modules.handedness.utils.graphics import read_dict_from_txt, plot_multiple_hands_from_dict

import re


def read_json_to_filtered_dataframe(json_file):
    """
    Reads a JSON file containing gloss data and extracts specific fields, returning a pandas DataFrame.
    
    :param json_file: Path to the JSON file.
    :return: Pandas DataFrame containing selected gloss data.
    """
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # List to store filtered data
    filtered_data = []
    
    for item in data:
        for gloss_id, gloss_data in item.items():
            # Create a dictionary with only the required fields
            filtered_entry = {
                'Gloss': gloss_data.get('Annotation ID Gloss: Dutch'),
                'Handedness': gloss_data.get('Handedness'),
                'Strong Hand': gloss_data.get('Strong Hand'),
                'Weak Hand': gloss_data.get('Weak Hand', None)  # Default to None if not present
            }
            filtered_data.append(filtered_entry)
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(filtered_data)
    
    return df


def main(df, pose_dir, output_dir):
    supported_handedness = ['1', '2s', '2a']
    # CLI ARGS
    parser = argparse.ArgumentParser(description="Process pose files and directories.")
    parser.add_argument('--handedness', nargs='+', help="List of handedness classes", default=supported_handedness)
    args = parser.parse_args()
    with open('/home/gomer/oline/PoseTools/src/modules/handedness/euclidean_model/reference_poses.pkl', 'rb') as file:
        reference_poses = pickle.load(file)

    pose_evaluator = EvaluatePoses(df, reference_poses, pose_dir)
    gloss_mapping = read_dict_from_txt('/home/gomer/oline/PoseTools/data/metadata/output/global_value_to_id.txt')
    

    df = read_json_to_filtered_dataframe(metadata)
    
    
    if '1' in args.handedness :
        output_file = os.path.join(output_dir, 'handedness_1.txt')
        L, R = 0, 0
        df_h1 = df[df['Handedness'] == '1']

        # Execute the appropriate function based on the arguments provided
        with open(output_file, 'w') as f_out:
            pose_evaluator = EvaluatePoses(df_h1, reference_poses)
            print("Processing handedness 1, n =", len(df_h1))
            filenames = set([file[:-6] for file in os.listdir(pose_dir) if file.endswith('.pkl')])
            for filename in filenames:
                cleaned_filename = re.sub(r'-\d+$', '', filename)
            
            
                # Check if the cleaned filename matches any gloss in the df_h1 DataFrame
                if cleaned_filename in df_h1['Gloss'].values:
                    # Extract the row corresponding to the matched gloss
                    row = df_h1[df_h1['Gloss'] == cleaned_filename]
                    
                    # Extract "Weak Hand" and "Strong Hand" values from the row
                    weak_hand = row['Weak Hand'].values[0]
                    strong_hand = row['Strong Hand'].values[0]
                    
                    # Get the reference pose for the strong hand
                    key = pose_evaluator.get_key_from_value(strong_hand)
            
                    if str(key) not in pose_evaluator.reference_poses.keys():
                        #print(f"Key not found for strong hand: {strong_hand, key}")
                        continue

                    reference_pose = pose_evaluator.reference_poses[str(key)]
                    
                    filepath = os.path.join(pose_dir, filename)
                    distance_L = pose_evaluator.get_distance(filepath + '-L.pkl', reference_pose)
                    distance_R = pose_evaluator.get_distance(filepath + '-R.pkl', reference_pose)

                    if distance_L < distance_R:
                        f_out.write(f"1, {filename}, {key}, {strong_hand}, L\n")
                        L += 1
                    elif distance_R < distance_L:
                        f_out.write(f"1, {filename}, {key}, {strong_hand}, R\n")
                        R += 1
                    else:
                        print(f"Error: {filename} has equal distances for both hands")	
                        exit()
                
        print(f"Total number of L files detected: {L}")
        print(f"Total number of R files detected: {R}")

        print(f"Total number of files detected: {L+R}")
   
                
    if '2s' in args.handedness :
        output_file = os.path.join(output_dir, 'handedness_2s.txt')
        L, R = 0, 0
        df_h2s = df[df['Handedness'] == '2s']

        # Execute the appropriate function based on the arguments provided
        with open(output_file, 'w') as f_out:
            pose_evaluator = EvaluatePoses(df_h2s, reference_poses)
            print("Processing handedness 2s, n =", len(df_h2s))
            filenames = set([file[:-6] for file in os.listdir(pose_dir) if file.endswith('.pkl')])
            for filename in filenames:
                cleaned_filename = re.sub(r'-\d+$', '', filename)
            
            
                # Check if the cleaned filename matches any gloss in the df_h1 DataFrame
                if cleaned_filename in df_h2s['Gloss'].values:
                    # Extract the row corresponding to the matched gloss
                    row = df_h2s[df_h2s['Gloss'] == cleaned_filename]
                    
                    # Extract "Weak Hand" and "Strong Hand" values from the row
                    weak_hand = row['Weak Hand'].values[0]
                    strong_hand = row['Strong Hand'].values[0]
                    # Get the reference pose for the strong hand
                    key = pose_evaluator.get_key_from_value(strong_hand)
            
                    if str(key) not in pose_evaluator.reference_poses.keys():
                        #print(f"Key not found for strong hand: {strong_hand, key}")
                        continue

                    reference_pose = pose_evaluator.reference_poses[str(key)]
                    
                    filepath = os.path.join(pose_dir, filename)
                    if os.path.exists(filepath + '-L.pkl'):
                        f_out.write(f"2s, {filename}, {key}, {strong_hand}, L\n")
                        L += 1
                    if os.path.exists(filepath + '-R.pkl'):
                        f_out.write(f"2s, {filename}, {key}, {strong_hand}, R\n")
                        R += 1
                
        print(f"Total number of L files detected: {L}")
        print(f"Total number of R files detected: {R}")
        
        print(f"Total number of files detected: {L+R}")

                    
    
    if '2a' in args.handedness:
        output_file = os.path.join(output_dir, 'handedness_2a.txt')
        L, R = 0, 0
        df_h2a = df[df['Handedness'] == '2a']

        # Execute the appropriate function based on the arguments provided
        with open(output_file, 'w') as f_out:
            pose_evaluator = EvaluatePoses(df_h2a, reference_poses)
            print("Processing handedness 2a, n =", len(df_h2a))
            filenames = set([file[:-6] for file in os.listdir(pose_dir) if file.endswith('.pkl')])
            for filename in filenames:
                cleaned_filename = re.sub(r'-\d+$', '', filename)
            
            
                # Check if the cleaned filename matches any gloss in the df_h2a DataFrame
                if cleaned_filename in df_h2a['Gloss'].values:
                    # Extract the row corresponding to the matched gloss
                    row = df_h2a[df_h2a['Gloss'] == cleaned_filename]
                    
                    # Extract "Weak Hand" and "Strong Hand" values from the row
                    weak_hand = row['Weak Hand'].values[0]
                    strong_hand = row['Strong Hand'].values[0]
                    
                    # Get the reference pose for the strong hand
                    key_strong = pose_evaluator.get_key_from_value(strong_hand)
                    key_weak = pose_evaluator.get_key_from_value(weak_hand)
            
                    if str(key_strong) not in pose_evaluator.reference_poses.keys() or str(key_weak) not in pose_evaluator.reference_poses.keys():
                        #print(f"Key not found for strong hand: {strong_hand, key}")
                        continue                    
                    
                    reference_pose_strong = pose_evaluator.reference_poses[str(key_strong)]
                    reference_pose_weak = pose_evaluator.reference_poses[str(key_weak)]
                
                    filepath = os.path.join(pose_dir, filename)
                    distance_L_s = pose_evaluator.get_distance(filepath + '-L.pkl', reference_pose_strong)
                    distance_L_w = pose_evaluator.get_distance(filepath + '-L.pkl', reference_pose_weak)
                    distance_R_s = pose_evaluator.get_distance(filepath + '-R.pkl', reference_pose_strong)
                    distance_R_w = pose_evaluator.get_distance(filepath + '-R.pkl', reference_pose_weak)
                    

                    if distance_L_s < distance_L_w:
                        f_out.write(f"2a, {filename}, {key_strong}, {strong_hand}, L\n")
                        L += 1
                    elif distance_L_w < distance_L_s:
                        f_out.write(f"2a, {filename}, {key_weak}, {weak_hand}, L\n")
                        L += 1
                    if distance_R_s < distance_R_w:
                        f_out.write(f"2a, {filename}, {key_strong}, {strong_hand}, R\n")
                        R += 1
                    elif distance_R_w < distance_R_s:
                        f_out.write(f"2a, {filename}, {key_weak}, {weak_hand}, R\n")
                        R += 1
                    
                
        print(f"Total number of L files detected: {L}")
        print(f"Total number of R files detected: {R}")
        print(f"Total number of files detected: {L+R}")
    
    for element in args.handedness:
        if element not in supported_handedness:
            print(f"Unsupported handedness class: {element}")
    


if __name__ == "__main__":

    directory_path = "../../../../mnt/fishbowl/gomer/oline/hamer_pkl"
    metadata = '/home/gomer/oline/PoseTools/data/metadata/glosses_meta.json'
    #output_file = "PoseTools/handedness/graphics/integrated_velocity_barplot.png"  # Output file for the bar plot
    output_dir = "/home/gomer/oline/PoseTools/results/handedness/hamer_pkl"  # Metadata file for the dataset
    

    

    main(metadata, directory_path, output_dir)
