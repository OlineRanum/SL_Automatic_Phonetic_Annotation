import os
import subprocess

# Input directories
input_directory = "/home/gomer/oline/PoseTools/src/modules/demo/demo_files/sentences"
output_directory = "/home/gomer/oline/PoseTools/src/modules/demo/demo_files/sentences/eaf_files"  # Directory where output .eaf files are stored

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# List all files in the input directory
files = os.listdir(input_directory + "/pose_files")

# Iterate over all .pose files
for file in files:
    print('Processing file:', file)
    if file.endswith(".pose"):
        # Extract the base file name (without extension)
        base_name = os.path.splitext(file)[0]

        # Define paths for the pose, elan, video, and output .eaf files
        pose_file = os.path.join(input_directory, f"pose_files/{base_name}.pose")
        elan_file = os.path.join(input_directory, f"{base_name}.eaf")
        video_file = os.path.join(input_directory, f"video_files/{base_name}.mp4")
        output_elan_file = os.path.join(output_directory, f"{base_name}.eaf")

        # Skip processing if the output .eaf file already exists
        if os.path.exists(output_elan_file):
            print(f"Skipping {base_name}: Output .eaf file already exists in {output_directory}")
            continue

        # Check if the corresponding .eaf and .mp4 files exist
        if os.path.exists(pose_file) and os.path.exists(video_file):
            # Construct the command to run
            cmd = f"pose_to_segments --pose=\"{pose_file}\" --elan=\"{output_elan_file}\" --video=\"{video_file}\" "

            # Run the command
            subprocess.run(cmd, shell=True)
            print(f"Processed: {base_name}")
        else:
            print(f"Skipping {base_name}: .eaf or .mp4 file is missing.")
