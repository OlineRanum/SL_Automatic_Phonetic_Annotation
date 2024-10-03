import json

def process_json_and_write_to_txt(json_file, output_file):
    """
    Reads the JSON file, filters instances where "Sign Type" is "2a", and writes the modified video IDs
    to a .txt file by removing the "-U" and appending ", R" for each relevant instance.
    
    Parameters:
    - json_file: The path to the input JSON file.
    - output_file: The path to the output .txt file.
    """
    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Open the output file for writing
    with open(output_file, 'w') as output_f:
        # Iterate through each entry in the JSON file
        for entry in data:
            # Iterate through each instance in the "instances" list
            for instance in entry['instances']:
                # Check if the "Sign Type" is "2a"
                if instance['Sign Type'] == "2a":
                    video_id = instance['video_id']
                    
                    # Check if the video ID ends with '-U'
                    if video_id.endswith('-U'):
                        # Remove the last two characters (-U)
                        modified_video_id = video_id[:-2]
                        
                        # Write the modified video ID in the required format
                        output_f.write(f"{modified_video_id}, R\n")
                        output_f.write(f"{modified_video_id}, L\n")

# Example usage
process_json_and_write_to_txt('PoseTools/data/metadata/metadata_1_2s_2a.json', 'PoseTools/data/metadata/output_2a.txt')
