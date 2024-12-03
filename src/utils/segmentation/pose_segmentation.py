import sys
import os

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

from PoseTools.processors.processors import HamerParser, EafParser, PoseLoader
from plotting_tools.hamer_to_gif import plot_landmarks_3d

def get_start_end_frame(start_time_ms, end_time_ms, sign_id):

    import cv2
    # Path to your video file
    video_file = '/home/gomer/oline/PoseTools/src/modules/demo/demo_files/sentences/video_files/'+sign_id+'.mp4'

    # Open the video file
    video = cv2.VideoCapture(video_file)

    # Get the frames per second (fps)
    fps = video.get(cv2.CAP_PROP_FPS)

    # Print the framerate
    print(f"Framerate of the video: {fps} fps")



    # Convert timestamps to frame numbers
    start_frame = int((start_time_ms / 1000) * fps)
    end_frame = int((end_time_ms / 1000) * fps)

    # Debugging: Print intermediate values
    print(f"Start time in seconds: {start_time_ms / 1000}")
    print(f"End time in seconds: {end_time_ms / 1000}")
    print(f"Start frame: {start_frame}")
    print(f"End frame: {end_frame}")

    # Release the video capture
    video.release()
    return start_frame, end_frame, fps

def log_failed_video(gloss, reason):
    log_file = "failed_videos.txt"
    with open(log_file, "a") as f:
        f.write(f"{gloss}\n")
    #print(f"Logged {gloss} to {log_file} with reason: {reason}")

def crop_video(start_frame, end_frame, sign_id):
    import cv2
    # Path to your video file
    #start_frame, end_frame, fps = get_start_end_frame(start_time_ms, end_time_ms, sign_id)
    
    
    video_file = '/home/gomer/oline/PoseTools/src/modules/demo/demo_files/sentences/video_files/'+sign_id+'.mp4'
    video = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS)

    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 0:
        #print(f"Error: Invalid frame count for {sign_id}")
        log_failed_video(sign_id,  "Invalid frame count")
        video.release()
        return 0

    print(f"Total frames in the video: {total_frames}")

    # Check if the frame range is valid
    if start_frame >= total_frames:
        print("Error: The frame range exceeds the total number of frames in the video.")
    else:
        if end_frame >= total_frames:
            end_frame = total_frames - 1
        # Set the starting frame position
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Define the video codec and create VideoWriter to save the cropped video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter('trimmed_videos/'+sign_id +'.mp4', fourcc, fps, (
            int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ))

        # Read and save frames from start to end frame
        for frame_num in range(start_frame, end_frame + 1):
            ret, frame = video.read()
            if not ret:
                print(f"Frame {frame_num} could not be read. Skipping.")
                continue
            output_video.write(frame)

        # Release everything
        video.release()
        output_video.release()
        print("Video cropped and saved as 'cropped_video.mp4'.")

def process_single_file(sign_id = '#O'):
    data_path = '../../../signbank_videos/'
    file_path = data_path + sign_id

    pose_path = '../../GMVISR/data/hamer/hamer_original/'
    pose_path = pose_path + sign_id

    # Load the ELAN file
    eaf_file = file_path + '.eaf'

    eaf_parser = EafParser(eaf_file)
    start_time_ms, end_time_ms = eaf_parser.parse_eaf()

    #crop_video(start_time_ms, end_time_ms, sign_id)

    start_frame, end_frame, fps = get_start_end_frame(start_time_ms, end_time_ms, sign_id)


    poseloader = HamerParser(pose_path + '.hamer')
    l_hand, r_hand = poseloader.read_hamer()
    #l_hand, r_hand = l_hand.squeeze(1), r_hand.squeeze(1)
    plot_landmarks_3d(l_hand, r_hand)
    print('Initial pose shape ', l_hand.shape)
    l_hand, r_hand = poseloader.temporal_segmentation(l_hand, start_frame, end_frame), poseloader.temporal_segmentation(r_hand, start_frame, end_frame)
    print('Cropped pose shape ', l_hand.shape)


import os, csv



def evaluate_pose_segmentation():
    # Run the check on the directory and save successful results as a CSV file
    output_file = "selected_signs.csv"
    error_percentage, error_count, total_files = check_eaf_files_in_directory(data_path, output_file)

    # Print the result
    print(f"Total files checked: {total_files}")
    print(f"Files with UnboundLocalError: {error_count}")
    print(f"Percentage of affected files: {error_percentage:.2f}%")


# Read the handshape_change_negative.csv file
def run_pose_segmentation():
    import pandas as pd
    df = pd.read_csv('../metadata/handshape_change_negative.csv')
    output_dir = '../../segmented_poses/'
    # Iterate through each row in the CSV and process the corresponding poses
    for index, row in df.iterrows():
        gloss = row['Gloss']  # Assuming 'Gloss' column exists
        start_frame = row['Start_Frame']
        end_frame = row['End_Frame']

        print(f"Processing {gloss} from frame {start_frame} to {end_frame}...")
        poseloader = PoseLoader(data_path + gloss + '.pose')
        data, conf = poseloader.read_keypoints_from_pose()
        data = data[start_frame:end_frame]
        conf = conf[start_frame:end_frame]
        
        poseloader.save_pose(data, conf, output_dir + gloss + '.pose')


# Read the handshape_change_negative.csv file
def trim_videos():
    import pandas as pd
    df = pd.read_csv('../metadata/handshape_change_negative.csv')
    output_dir = 'trimmed_videos/'
    # Iterate through each row in the CSV and process the corresponding poses
    for index, row in df.iterrows():
        gloss = row['Gloss']  # Assuming 'Gloss' column exists
        start_frame = row['Start_Frame']
        end_frame = row['End_Frame']

        # Construct the expected output file path
        output_video_path = os.path.join(output_dir, f"{gloss}.mp4")

        # Check if the video already exists
        if os.path.exists(output_video_path):
            continue
            #print(f"Skipping {gloss} - video already exists at {gloss}")
        else:
            print(f"Processing {gloss} from frame {start_frame} to {end_frame}...")
            try:
                crop_video(start_frame, end_frame, gloss)
            except:
                crop_video(start_frame, end_frame, gloss)



def check_eaf_files_in_directory(directory, output_file):
    # Get a list of all .eaf files in the directory
    eaf_files = [f for f in os.listdir(directory) if f.endswith('.eaf')]

    total_files = len(eaf_files)
    error_count = 0

    # Open the output CSV file to write successful results
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Gloss", "Start_Frame", "End_Frame"])  # Write the CSV header

        for eaf_file in eaf_files:
            try:
                # Create the file path for the .eaf file
                file_path = os.path.join(directory, eaf_file)
                gloss = os.path.splitext(eaf_file)[0]  # Assuming gloss is derived from filename
                
                # Parse the EAF file
                eaf_parser = EafParser(file_path)
                start_time_ms, end_time_ms = eaf_parser.parse_eaf()

                # Get the start and end frame
                start_frame, end_frame, fps = get_start_end_frame(start_time_ms, end_time_ms, gloss)

                # Write the gloss and frame info to the CSV file
                csvwriter.writerow([gloss, start_frame, end_frame])

            except UnboundLocalError:
                # If an UnboundLocalError occurs, count it
                print(f"UnboundLocalError found in file: {eaf_file}")
                error_count += 1
            except Exception as e:
                # Handle other exceptions if necessary
                print(f"An error occurred with file {eaf_file}: {e}")
    
    # Calculate the percentage of files that threw the UnboundLocalError
    if total_files > 0:
        error_percentage = (error_count / total_files) * 100
    else:
        error_percentage = 0

    return error_percentage, error_count, total_files

if __name__ == '__main__':
    # Define the paths
    data_path = '../../../signbank_videos/'
    check_eaf_files_in_directory(data_path, 'timestamps.txt')