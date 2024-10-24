import os
import json

# Function to check if a corresponding .pkl file exists
def check_pkl_exists(directory, video_id):
    pkl_filename = f"{video_id}.pkl"
    return os.path.exists(os.path.join(directory, pkl_filename))

# Function to process the JSON file and check for missing .pkl files
def process_file(file_path, directory):
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
                pkl_exists = check_pkl_exists(directory, video_id)
                if not pkl_exists:
                    counter += 1
                    missing_glosses.append({
                        "gloss": gloss,
                        "video_id": video_id
                    })
    
    
    return missing_glosses, counter


# Example usage
json_file_path = 'PoseTools/data/metadata/output/handedness_1_2/metadata_wlasl.json'
pkl_directory = 'PoseTools/data/datasets/wlasl_small'

# Process the file and get missing glosses
missing_glosses, counter = process_file(json_file_path, pkl_directory)

# Output missing glosses
for gloss_info in missing_glosses:
    print(f"Gloss: {gloss_info['gloss']}, Video ID: {gloss_info['video_id']} is missing .pkl file.")

print('Total missing glosses:', counter)