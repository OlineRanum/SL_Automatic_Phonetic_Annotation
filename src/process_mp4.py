import os
import argparse, subprocess,shutil

from PoseTools.data.parsers_and_processors.converters.vid_to_hamer import main_hamer
from PoseTools.src.modules.old_segmentation.segmentation import main_activation
from PoseTools.src.modules.handedness.utils.eval_1_2_hands import main_handedness
from PoseTools.src.modules.orientation.orientation import main_orientation
from PoseTools.src.modules.location.location import main_location
from PoseTools.src.modules.visualization.visualize import main_visualization
from PoseTools.src.modules.base.base import DataModule
from PoseTools.src.modules.activation.activation import main_activation
from PoseTools.data.parsers_and_processors.converters.vid_to_hamer import main_hamer        
from PoseTools.src.modules.handshapes.engines.ed_algorithm.handshape import main_handshape



def run_videos_to_poses(directory, format="mediapipe"):
    """
    Runs the 'videos_to_poses' command line tool with the specified directory and format.
    """
    input_folder = directory + '/video_files'
    output_folder = directory + '/pose_files'
    cmd = ["videos_to_poses", "--format", format, "--directory", input_folder]
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)  # Print standard output
        move_pose_files(input_folder, output_folder)
    except subprocess.CalledProcessError as e:
        print(f"Error while running videos_to_poses: {e}")
        print(f"Standard Output: {e.stdout}")
        print(f"Standard Error: {e.stderr}")

def move_pose_files(source_dir, destination_dir):
    """
    Moves all files ending with .pose from the source directory to the destination directory.

    Args:
    - source_dir (str): Path to the source directory.
    - destination_dir (str): Path to the destination directory.
    """
    # Ensure the destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print(f"Created destination directory: {destination_dir}")

    # Loop through all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith(".pose"):  # Check if the file ends with .pose
            source_path = os.path.join(source_dir, filename)
            destination_path = os.path.join(destination_dir, filename)
            
            # Move the file
            shutil.move(source_path, destination_path)
            print(f"Moved: {source_path} -> {destination_path}")

def get_repository_data(input_folder):
    base_filenames = [filename[:-4] for filename in os.listdir(input_folder + '/video_files') if filename.endswith('.mp4')]
    print('\nAll files: ')
    for base_filename in base_filenames:
        print(base_filename)   
    return base_filenames

def main(args, conver_data = False):
    """
    Main script to convert data and process labels.
    """
    
    input_folder, output_folder = args.input_folder, args.output_folder

    # Convert data to poses
    if conver_data:
        # Convert to Hamer
        try:
            print('Converting to HaMer Hanshape poses...\n')
            main_hamer(input_folder)
        except Exception as e:
            print(f"Error while running main_hamer: {e}")
            print("Ensure that the HAMER server is running.")
        
        # Convert to Mediapipe
        try:
            print('\nConverting to Mediapipe full-body poses...')
            run_videos_to_poses(input_folder)
        except Exception as e:
            print(f"Error while running pose converter: {e}")
            
    # Get repository data
    base_filenames = get_repository_data(input_folder)

    # Process labels
    for base_filename in base_filenames:
        print('\nProcessing: ', base_filename)
        
        data = DataModule(base_filename, base_dir = input_folder, args= args)   
        

        gif_path = '/home/gomer/oline/PoseTools/src/server/public/gifs/'+base_filename+'.gif'
        boolean_activity_arrays, sign_activity_arrays, start, stop = main_activation(data, n_components = 2)
        
        boolean_activity_arrays, sign_activity_arrays = data.select_data(start, stop, skip = 5, boolean_activity_arrays = boolean_activity_arrays, sign_activity_arrays = sign_activity_arrays) 

        # TODO - setup for loading data with new base class
        handedness = main_handedness(boolean_activity_arrays)
        
        handshape_predictions= main_handshape(data, input_folder + '/hamer_pkl', output_folder, boolean_arrays=boolean_activity_arrays, base_filename='normalized_' +base_filename, args=args)
        
        locations = main_location(data, print_results = False)
        orientations = main_orientation(data, print_results=False)
        

        main_visualization(data, save_anim_path=gif_path, 
                        boolean_activity_arrays=boolean_activity_arrays,
                        sign_activity_arrays=sign_activity_arrays, 
                        handshapes = handshape_predictions, 
                        handedness = handedness, 
                        orientations = orientations, 
                        locations = locations)


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process video files with HAMER and related tools.")
    parser.add_argument(
        "--input_folder", 
        type=str, 
        required=True, 
        help="Path to the input folder containing video files or data."
    )
    parser.add_argument(
        "--output_folder", 
        type=str, 
        default="/home/gomer/oline/PoseTools/src/modules/server/demo_files/graphics/handshapes",
        help="Path to the output folder for processed files."
    )
    parser.add_argument(
        "--subsample_index", 
        type=int, 
        nargs='+',   
        choices=range(0, 21),
        default=None,
        help="A list of hamer keypoint indexes to subsample"
    )
    parser.add_argument(
        "--feature_transformation", 
        type=str, 
        choices=['orientations', 'pdm'], 
        default='pdm',
        help="A list of hamer keypoint indexes to subsample"
    )
    parser.add_argument(
        "--subsample_finger",
        nargs='+',
        choices=['thumb', 'index', 'middle', 'ring', 'pinky'],  
        default=['thumb', 'index', 'middle', 'ring', 'pinky'],
        help="A list of hamer keypoint indexes to subsample"
    )
    parser.add_argument(
        "--mask_type", 
        type=str, 
        default=None,
        choices=["gomer", "wrist_to_fingers"],
        help="Type of mask to apply to the PDM."
    )
        
    args = parser.parse_args()
    
    # Run the main script
    main(args)

