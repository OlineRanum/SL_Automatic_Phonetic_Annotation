import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend if running on a server
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm
import os
from src.modules.data.mocap_data.utils.dataloader import DataLoader
from src.modules.data.mocap_data.utils.normalizer import Normalizer
import re

def process_file(file_path, frames_to_skip=10, fps=15):
    """
    Processes a single file and generates frames and an animation.
    
    Args:
        file_path (str): Path to the input CSV file.
        frames_dir (str): Directory to save frames.
        gifs_dir (str): Directory to save GIFs.
        frames_to_skip (int): Number of frames to skip when processing.
        fps (int): Frames per second for the animation.
    """

    # Extract name from filename
    filename = os.path.basename(file_path)



    
    loader = DataLoader(file_path, mode='fullpose')
    hand_edges = loader.prepare_hand_data()
    normalizer = Normalizer(loader)

    # Load and preprocess data
    df = loader.load_data()


    normalizer.load_transformations()
    right_hand, marker_names_hands = loader.get_hand()
    right_hand = right_hand[::frames_to_skip]
    normalized_right_handshape = normalizer.normalize_handshape(right_hand, marker_names_hands)

    print(f"Processing file: {filename}")
    print(f"shape of data ", normalized_right_handshape.shape)




def main(data_list):

    
    data_folder = '/home/oline/SL_Automatic_Phonetic_Annotation/src/server/public/data/mocap/'

    for file in data_list:
        print(f"Processing file: {file}", flush = True)
        file_path = os.path.join(data_folder, file)
        process_file(file_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process MoCap files")
    parser.add_argument('--data_list', nargs='+', help="List of data files to process", required=True)
    args = parser.parse_args()

    # Pass the `data_list` to the main function
    main(data_list=args.data_list)