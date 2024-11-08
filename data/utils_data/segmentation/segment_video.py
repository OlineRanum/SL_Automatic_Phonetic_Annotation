import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# Parameters
input_video = "/home/gomer/oline/PoseTools/data/datasets/test_data/M20241024_4718.mp4"  # Path to your main video
input_txt = "/home/gomer/oline/PoseTools/data/datasets/test_data/timestamps.txt"     # Path to your timestamps text file
fps = 60                         # Frames per second of the input video

# Helper function to convert time to seconds
def time_to_seconds(time_str):
    h, m, s = time_str.split(':')
    s, ms = s.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

# Read the timestamps and signs from the text file
segments = []
with open(input_txt, 'r') as file:
    lines = file.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.isdigit():  # Skip the numbering lines
            i += 1
            continue
        
        time_range = lines[i].strip()
        sign_name = lines[i + 1].strip()

        # Add error handling to ensure proper format
        try:
            start_time_str, end_time_str = time_range.split(' --> ')
            start_time = time_to_seconds(start_time_str)
            end_time = time_to_seconds(end_time_str)
            segments.append((sign_name, start_time, end_time))
            i += 3  # Move to the next segment
        except ValueError:
            print(f"Skipping malformed line: {time_range}")
            i += 1
            continue

# Process each segment and save it as an individual file
for sign_name, start_time, end_time in segments:
    # Calculate the last 2/5 of the time segment
    segment_duration = end_time - start_time
    adjusted_start_time = start_time + (3 * segment_duration / 5)  # Start at 3/5 into the segment
    adjusted_end_time = end_time

    output_filename = f"/home/gomer/oline/PoseTools/data/datasets/test_data/segmented/{sign_name}.mp4"
    ffmpeg_extract_subclip(input_video, adjusted_start_time, adjusted_end_time, targetname=output_filename)
    print(f"Created segment: {output_filename}")
