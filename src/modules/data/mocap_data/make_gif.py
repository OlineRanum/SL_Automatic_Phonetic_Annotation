import numpy as np
import os

# Imports from your local modules
from dataloader import DataLoader
from normalizer import Normalizer

from visualize_reference_data import NewPlotter 

# 1) Setup the file path to your CSV
file_path = '/home/oline/3D_MoCap/data/V_markerData.csv'

# 2) Create the loader and read the dataframe
loader = DataLoader(file_path, mode='fullpose')
df = loader.load_data()

# Optionally figure out how many frames are in the dataset
# (Adjust this logic to match how your DataLoader organizes frames)
num_frames = df['Frame'].nunique()  # for example, if 'Frame' column exists
num_frames = 10
# You might also have: num_frames = loader.num_frames, depending on your DataLoader

# 3) Create Normalizer (to get the same logic as in your example)
normalizer = Normalizer()

# 4) Pre-allocate a NumPy array for body_data
#    We'll assume each frame has the same marker count. 
#    The shape must be (num_frames, num_body_points, 3).
#    If you do not know the exact number of markers, you can:
#        - read one frame to see how many markers
#        - or let your DataLoader provide it

frame0_marker_names, frame0_marker_data = loader.get_marker_data(1)
num_body_points = len(frame0_marker_data)
body_data = np.zeros((num_frames, num_body_points, 3), dtype=np.float32)

# 5) Fill the body_data array frame by frame
for frame_idx in range(num_frames):
    marker_names, marker_data = loader.get_marker_data(frame_idx*100+1)
    marker_data_normalized = normalizer.full_pose_normalizer(marker_names, marker_data)
    body_data[frame_idx, :, :] = marker_data_normalized

# 6) [Optional] If you already have right-hand data or right-wrist data from somewhere,
#    load or compute them here; for now, let's create simple placeholders:
right_hand_data = np.random.rand(num_frames, 21, 3)  # Replace with real hand data
right_wrist_data = np.linspace(0, 10, num_frames) + np.random.randn(num_frames)

# 7) Now create your NewPlotter instance with real body_data
frames_dir = "./frames_example"
save_video_path = "example_animation_fullpose.mp4"

plotter = NewPlotter(
    body_data=body_data,               # from CSV
    right_hand_data=right_hand_data,   # your actual right-hand data
    right_wrist_data=right_wrist_data, # e.g. velocity or position data
    frames_dir=frames_dir,
    save_frames=True
)

# 8) Generate the animation
plotter.create_animation(save_path=save_video_path, fps=10)
