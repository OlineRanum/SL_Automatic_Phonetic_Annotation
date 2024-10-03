import argparse
import os 
from PoseTools.handedness.utils.processor_h1 import process_pose_file, process_directory


def main(directory_path, output_file, filtered_file_path):
    supported_handedness = ['1', '2s', '2a']
    # CLI ARGS
    parser = argparse.ArgumentParser(description="Process pose files and directories.")
    parser.add_argument('--handedness', nargs='+', help="List of handedness classes", default=supported_handedness)
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Execute the appropriate function based on the arguments provided
    if '1' in args.handedness :
        print("Processing handedness 1")
        process_directory(directory_path, output_file, filtered_file_path, method = 'detect_LR')
            
    if '2s' in args.handedness :
        print("Processing handedness 2s")
    
    if '2a' in args.handedness :
        print("Processing handedness 2a")
    
    for element in args.handedness:
        if element not in supported_handedness:
            print(f"Unsupported handedness class: {element}")
    


if __name__ == "__main__":

    directory_path = "../signbank_videos/"  # Path to the directory containing .pose files
    output_file = "PoseTools/handedness/graphics/integrated_velocity_barplot.png"  # Output file for the bar plot
    filtered_file_path = "pose/metadata/filtered_output_reduced.txt"  # Metadata file for the dataset

    main(directory_path, output_file, filtered_file_path)
    