import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.dataloader import DataLoader
from plotter import Plotter
from utils.normalizer import Normalizer

# File path to the CSV file
file_path = '/home/gomer/oline/3D_MoCap/data/9_markerData.csv'

# Load the data
loader = DataLoader(file_path, mode = 'fullpose')
df = loader.load_data()

# Specify the frame to plot
frame_to_plot = 600

# Get marker data for the specified frame
marker_names, marker_data = loader.get_marker_data(frame_to_plot)

# Plot the marker data
plot_in_3d = True  # Set to False to plot in 2D

if plot_in_3d:
    plotter = Plotter(loader.edges)
    normalizer = Normalizer()
    marker_data = normalizer.full_pose_normalizer(marker_names, marker_data)
    
    savepath = 'figs/3d_markers_fullpose.png'
    plotter.plot_3d_markers(marker_names, marker_data, angle_azim = -60, angle_elev=15, group = 'fullpose', save_path=savepath)
