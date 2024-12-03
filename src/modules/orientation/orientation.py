import numpy as np
import os 
from pose_format import Pose
from PoseTools.src.utils.preprocessing import PoseSelect, PoseNormalize
import pickle
from PoseTools.src.utils.plotting.hamer import plot_hamer_hand_3d, plot_gif, plot_gif_with_normals
from PoseTools.src.modules.base.base import DataModule

class Orientation(DataModule):
    def __init__(self, data):
        self.data = data
        self.hamer_right = data.hamer_right
        self.hamer_left = data.hamer_left
        self.normal_vectors_left = data.normal_vectors_left
        self.normal_vectors_right = data.normal_vectors_right


    def calculate_palm_normal(self, wrist, index_base, pinky_base):
        # Calculate vectors
        v1 = index_base - wrist
        v2 = pinky_base - wrist
        
        # Cross product to find the normal vector
        normal = np.cross(v1, v2)
        
        # Normalize the normal vector
        normal_unit = normal / np.linalg.norm(normal)
        
        return normal_unit
    
    def calculate_relative_orientation(self, target_vector):
        """
        Compute relative azimuth and elevation angles between the reference vector and the target vector.

        Args:
        - reference_vector: A numpy array of shape (3,) representing the reference vector.
        - target_vector: A numpy array of shape (3,) representing the target vector.

        Returns:
        - azimuth: Relative horizontal rotation angle in degrees.
        - elevation: Relative vertical rotation angle in degrees.
        """
        x, y, z = target_vector
        # Normalize the input to ensure it's on the unit sphere
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        x, y, z = x / magnitude, y / magnitude, z / magnitude

        # Calculate theta (polar angle)
        theta = np.arccos(z)

        # Calculate phi (azimuthal angle)
        phi = np.arctan2(y, x)

        # Ensure phi is in the range [0, 2*pi)
        if phi < 0:
            phi += 2 * np.pi
        
        theta = np.degrees(theta)
        phi = np.degrees(phi)

        return theta, phi


    def rotate_to_align_with_z(self, vectors):
        """
        Rotate a collection of 3D vectors so that the first vector is aligned with [0, 0, 1].

        Args:
            vectors: List or numpy array of shape [N, 3] representing 3D vectors.

        Returns:
            rotated_vectors: Numpy array of rotated vectors with the first vector aligned with [0, 0, 1].
        """
        from scipy.spatial.transform import Rotation as R
        # Convert to numpy array if not already
        vectors = np.array(vectors)
        
        # Validate dimensions
        if vectors.ndim != 2 or vectors.shape[1] != 3:
            raise ValueError("Input vectors must be a 2D array with shape [N, 3].")
        
        # Define target vector
        target_vector = np.array([0.0, 0.0, 1.0])

        # Get and normalize the first vector
        first_vector = vectors[0]
        norm_first = np.linalg.norm(first_vector)
        if norm_first < 1e-8:
            raise ValueError("The first vector has near-zero magnitude and cannot define a rotation.")
        first_vector_normalized = first_vector / norm_first

        # Compute cross product and dot product
        rotation_axis = np.cross(first_vector_normalized, target_vector)
        axis_length = np.linalg.norm(rotation_axis)
        dot_product = np.dot(first_vector_normalized, target_vector)

        # Handle special cases
        if axis_length < 1e-10:
            if dot_product > 0:
                # Vectors are already aligned
                return vectors
            else:
                # Vectors are opposite; choose an arbitrary orthogonal rotation axis (e.g., x-axis)
                rotation_axis = np.array([1.0, 0.0, 0.0])
                rotation_angle = np.pi  # 180 degrees
        else:
            # Normalize rotation axis
            rotation_axis_normalized = rotation_axis / axis_length
            # Compute rotation angle
            rotation_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

        # Create rotation object using scipy
        rotation = R.from_rotvec(rotation_angle * rotation_axis_normalized if axis_length >= 1e-10 else rotation_angle * rotation_axis)
        rotation_matrix = rotation.as_matrix()

        # Apply rotation to all vectors
        rotated_vectors = vectors @ rotation_matrix.T

        return rotated_vectors

    def categorize_named_zones(self, unit_vector, side):
        """
        Categorize azimuth and elevation into named zones.
        """
        x, y, z = unit_vector

        # Determine the octant based on the signs of x, y, and z
        zone_name = 'Palm'

        # TODO: We need to determin the middle pose line using the pose operator 

        if z >= 0.4 :
            zone_name += " Back-"
        elif z <= -0.4:
            zone_name += "Front-"
        else:
            zone_name += "Side-"
        if y >= 0.4:
            zone_name += "Down-"
        if y <= -0.4:
            zone_name += "Up-"
    
        if x >= 0.6:
            zone_name += "In-"
        if x <= -0.6:
            zone_name += "Out-"
        try:
            return zone_name[:-1]
        except:
            return "None"
        
    def angular_zone(self, theta, phi ):
        """
        Categorize azimuth and elevation into named zones.
        """
        zone_name = ''

        if 0 < theta <=  90 :
            zone_name += " Back-"
        elif 180 < theta <=  270 :
            zone_name += "Front-"
        else:
            zone_name += "Side-"

        try:
            return zone_name[:-1]
        except:
            return "None"
        
    def calculate_relative_angles(self, side="right"):
        if side == "right":
            hamer_data = self.hamer_right
            normal_vectors = self.normal_vectors_right
        elif side == "left":
            hamer_data = self.hamer_left
            normal_vectors = self.normal_vectors_left
        else:
            raise ValueError("Invalid side specified. Use 'left' or 'right'.")


        #normal_vectors = self.rotate_to_align_with_z(normal_vectors)
        
        

        # Compute absolute azimuth and elevation
        theta_array = []
        phi_array = []

        frame = 0
        for vector in normal_vectors:
            theta, phi = self.calculate_relative_orientation(vector)
            theta_array.append(theta)
            phi_array.append(phi)


        theta_array = np.array(theta_array)
        phi_array = np.array(phi_array)

        # Handle azimuth wraparound
        #azimuth_diff = (azimuth_array[1:] - azimuth_array[:-1] + 180) % 360 - 180
        #smooth_azimuth = np.concatenate([[azimuth_array[0]], np.cumsum(azimuth_diff)])

        # Smooth the angles
        #smoothed_azimuth = gaussian_filter1d(smooth_azimuth, sigma=2)
        #smoothed_elevation = gaussian_filter1d(elevation_array, sigma=2)

        relative_angles = []
        for i in range(len(normal_vectors)):
            relative_theta = theta_array[i] 
            relative_phi = phi_array[i] 
            
            zone_name = self.categorize_named_zones(normal_vectors[i], side)
            #zone_name_angular  = self.angular_zone(relative_theta, relative_phi)
            relative_angles.append({
                "frame": i,
                "theta": relative_theta,
                "phi": relative_phi,
                "zone": zone_name
            })
        
        return relative_angles, normal_vectors
    
    def get_zones(self, relative_angles, normal_vectors, print_results = False): 
            
        theta, phi = [], []
        
        zones = [zone['zone'] for zone in relative_angles]
        
        for angle_info, vector in zip(relative_angles, normal_vectors):
            norm = np.round(np.linalg.norm(vector),2)
            
            if print_results:
                
                print(f"Frame {angle_info['frame'] }: [{vector[0]:.2f}, {vector[1]:.2f}, {vector[2]:.2f}], norm = {norm}, Theta = {angle_info['theta']:.2f}°, "
                    f"phi = {angle_info['phi']:.2f}°, Zone = {angle_info['zone']}")
            theta.append(np.round(angle_info['phi'],2))
            phi.append(np.round(angle_info['theta'],2))
        theta, phi = np.array(theta), np.array(phi)
        #plot_normal_vectors(normal_vectors[::3], azim, ele)
        
        #plot_hamer_hand_3d(self.hamer_right[time_index], self.base_filename, normal_vectors[self.time_index])
        return zones

import matplotlib.pyplot as plt 
import imageio
def plot_normal_vectors(normal_vectors, azim, ele, gif_filename="/home/gomer/oline/PoseTools/src/modules/orientation/normal_vectors.gif", frame_interval=1000):
    """
    Plots a series of normal vectors iteratively and generates a GIF.

    Args:
        normal_vectors: List of numpy arrays, each representing a normal vector (shape: (3,)).
        gif_filename: The filename for the output GIF.
        frame_interval: Interval in milliseconds between frames in the GIF.
    """
    # Ensure the vectors are numpy arrays
    normal_vectors = np.array(normal_vectors)

    # Initialize the GIF frame list
    frames = []

    # Loop through normal vectors and plot each frame
    for i, vector in enumerate(normal_vectors):
        # Create a new figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color='r', length=1, normalize=True)

        # Set plot limits
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        # Set labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Normal Vector\nFrame {i}, A = {azim[i]}, E = {ele[i]}\n{vector}")
        ax.view_init(elev=90, azim=90)
        # Save the current frame as an image
        plt.savefig(f"/home/gomer/oline/PoseTools/src/modules/orientation/frames/frame_{i}.png")
        plt.close(fig)

        # Add the frame to the list
        frames.append(imageio.imread(f"/home/gomer/oline/PoseTools/src/modules/orientation/frames/frame_{i}.png"))

    # Generate the GIF
    imageio.mimsave(gif_filename, frames, duration=frame_interval / 1000.0, loop = 0)

    # Clean up temporary image files
    #for i in range(len(normal_vectors)):
        #try:
            #os.remove(f"/home/gomer/oline/PoseTools/src/modules/orientation/frames/frame_{i}.png")
        #except FileNotFoundError:
        #    pass
    print(f"GIF saved as {gif_filename}")

from scipy.ndimage import gaussian_filter1d

def smooth_vectors_gaussian(vectors, sigma=1):
    """
    Smooth normal vectors using a Gaussian filter.

    Args:
        vectors: List of numpy arrays, each representing a normal vector (shape: (3,)).
        sigma: Standard deviation for Gaussian kernel.

    Returns:
        smoothed_vectors: List of smoothed vectors.
    """
    vectors = np.array(vectors)
    smoothed_vectors = []

    # Apply Gaussian filter to each dimension (x, y, z)
    for dim in range(vectors.shape[1]):
        smoothed_dim = gaussian_filter1d(vectors[:, dim], sigma=sigma)
        smoothed_vectors.append(smoothed_dim)

    smoothed_vectors = np.stack(smoothed_vectors, axis=1)

    # Re-normalize each vector
    smoothed_vectors /= np.linalg.norm(smoothed_vectors, axis=1, keepdims=True)

    return smoothed_vectors




def main_orientation(data, print_results = False, return_angles = False):
    locations = []
    normals = []
    angles = []
    for side in ["right", "left"]:
        
        analyzer = Orientation(data)
        # Compute relative angular changes for the right hand
        relative_angles, normal_vectors = analyzer.calculate_relative_angles(side=side)
        zones = analyzer.get_zones(relative_angles, normal_vectors, print_results)
        locations.append(zones)
        normals.append(normal_vectors)
        angles.append(relative_angles)
        #print(side)
        #idx = 0
        #for i in zip(normal_vectors, relative_angles):
            #idx += 1
            #print(idx ,i[0], i[1], np.abs(np.round(np.sum(i[0] - normal_vectors[0]),2)))
            #print(idx, int(i[1]['phi']), int(i[1]['theta']), i[1]['zone'])
    #plot_gif_with_normals(data.hamer_left,data.hamer_right, normals[0], normals[1], gif_path='/home/gomer/oline/PoseTools/src/modules/orientation/handshapes.gif')
        
    if return_angles:
        return locations, angles, normals
    else:
        return locations




