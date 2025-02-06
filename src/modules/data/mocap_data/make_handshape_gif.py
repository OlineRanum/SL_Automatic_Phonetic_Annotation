import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from mpl_toolkits.mplot3d import Axes3D
from dataloader import DataLoader
from normalizer import Normalizer
from plotter import Plotter


if __name__ == "__main__":
    file_path = '/home/oline/3D_MoCap/data/V_markerData.csv'
    loader = DataLoader(file_path)
    df = loader.load_data()

    reference_frame = 1
    marker_names, marker_data = loader.get_marker_data(reference_frame)

    normalizer = Normalizer()
    translation, scale_factor, rotation_matrix = normalizer.compute_normalization_transform(marker_names, marker_data)
    normalized_ref_data = normalizer.apply_normalization(marker_data, marker_names, translation, scale_factor, rotation_matrix)

    riwr_idx = marker_names.tolist().index("RIWR")
    rowr_idx = marker_names.tolist().index("ROWR")
    print("Reference frame RIWR:", normalized_ref_data[riwr_idx])
    print("Reference frame ROWR:", normalized_ref_data[rowr_idx])

    frame_to_plot = 5000
    marker_names_test, marker_data_test = loader.get_marker_data(frame_to_plot)
    normalized_marker_data = normalizer.apply_normalization(marker_data_test, marker_names_test, translation, scale_factor, rotation_matrix)


    plotter = Plotter(loader.edges)
    plotter.plot_3d_markers(marker_names_test, normalized_marker_data, angle_elev=30, angle_azim=30, save_path='figs/normalized_view.png')

    # Create a GIF from the normalized frames
    plotter.create_gif_from_frames(df, 'figs/3d_markers_hands.gif', start_frame=0, end_frame=6000, step=50,
                                   translation=translation, scale_factor=scale_factor, rotation_matrix=rotation_matrix)