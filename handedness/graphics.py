import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

def plot_velocity(velocity_r, velocity_l, pose_filename):
    # Calculate the magnitude of the velocity vectors
    vel_r_mag = np.sqrt(np.sum(velocity_r**2, axis=1))  # Use np.sqrt and sum for norm
    vel_l_mag = np.sqrt(np.sum(velocity_l**2, axis=1))

    # Create a 1D plot of the velocity profiles
    frames = np.arange(len(vel_r_mag))

    plt.figure(figsize=(10, 5))
    plt.plot(frames, vel_r_mag, label="Right Hand Velocity")
    plt.plot(frames, vel_l_mag, label="Left Hand Velocity", linestyle='--')

    plt.xlabel("Frame")
    plt.ylabel("Velocity Magnitude")
    plt.title("Velocity Profile of Both Hands")
    plt.legend()
    plt.grid(True)
    plt.savefig('PoseTools/handedness/graphics/velocity_'+pose_filename+'.png')


def plot_integrated_velocities(integrated_velocities, output_file):
    # Create bar plot for integrated velocities
    n_files = len(integrated_velocities)
    
    # Separate right and left hand data
    integrated_r = [item[0] for item in integrated_velocities]
    integrated_l = [item[1] for item in integrated_velocities]
    
    # Generate bar plot
    x = np.arange(n_files)  # X-axis values (just indices)
    
    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, integrated_r, width=0.4, label="Right Hand", color="red")
    plt.bar(x + 0.2, integrated_l, width=0.4, label="Left Hand", color="blue")
    
    plt.ylabel("Integrated Velocity")
    plt.title("Integrated Velocity of Both Hands across Files")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(output_file)
    plt.close()

def plot_position(pos_r, pos_l, pose_filename):
    # Calculate the magnitude of the velocity vectors

    # Create a 1D plot of the velocity profiles
    frames = np.arange(len(pos_r))

    plt.figure(figsize=(10, 5))
    plt.scatter(frames, pos_r, label="Right Hand Velocity")
    plt.scatter(frames, pos_l, label="Left Hand Velocity", linestyle='--')

    plt.xlabel("Frame")
    plt.ylabel("Position Magnitude")
    plt.title("Position Profile of Both Hands")
    plt.legend()
    plt.grid(True)
    plt.savefig('PoseTools/handedness/graphics/position_'+pose_filename+'.png')

