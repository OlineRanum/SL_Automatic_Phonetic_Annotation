import argparse
import os 
from PoseTools.src.modules.handedness.utils.processor_h1 import process_pose_file, process_directory, process_directory_h1
import json
import pandas as pd

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

    df = read_json_to_filtered_dataframe(metadata)
    print(df.head())

    # Execute the appropriate function based on the arguments provided
    if '1' in args.handedness :
        df_h1 = df[df['Handedness'] == '1']
        print("Processing handedness 1")
        output_file = os.path.join(output_dir, 'handedness_1.txt')
        process_directory_h1(df_h1, pose_dir, output_file)
            
    if '2s' in args.handedness :
        print("Processing handedness 2s")
    
    if '2a' in args.handedness :
        print("Processing handedness 2a")
    
    for element in args.handedness:
        if element not in supported_handedness:
            print(f"Unsupported handedness class: {element}")
    



if __name__ == "__main__":

    directory_path = "../../../../mnt/fishbowl/gomer/oline/hamer_pkl"
    metadata = '/home/gomer/oline/PoseTools/data/metadata/glosses_meta.json'
    #output_file = "PoseTools/handedness/graphics/integrated_velocity_barplot.png"  # Output file for the bar plot
    output_dir = "/home/gomer/oline/PoseTools/results/handedness/hamer_pkl"  # Metadata file for the dataset
    

    

    main(metadata, directory_path, output_dir)
    