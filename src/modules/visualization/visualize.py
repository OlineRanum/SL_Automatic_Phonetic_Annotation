import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import cv2
from tqdm import tqdm
import os, glob
import traceback
from matplotlib.animation import FuncAnimation, FFMpegWriter


from PoseTools.src.modules.handedness.utils.graphics import PosePlotter

from PoseTools.src.modules.features.feature_transformations import DistanceFeatures, MaskFeatures

import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend for rendering

class PhoneticAnnotationPlot:
    def __init__(self, data,  sign_activity_arrays, boolean_activity_arrays, handshapes=None, handedness=None, orientations = None, locations = None, save_frames= True):
        # Initialization code
        self.feature_extractor = DistanceFeatures()
        self.masker = MaskFeatures(mask_type='gomer')

        self.base_path = data.BASE_DIR
        self.left_wrist_activity, self.right_wrist_activity = sign_activity_arrays

        self.left_wrist_activity_b, self.right_wrist_activity_b = boolean_activity_arrays
        
        self.video_path = data.video_path
        

        self.wrist_left = data.wrist_left
        self.wrist_right = data.wrist_right

        self.base_filename = data.base_filename
        self.save_frames = save_frames

        self.handedness = handedness

        self.handshapes = handshapes[0]
        self.handshapes_top3 = handshapes[1]

        self.orientations_l = orientations[0]
        self.orientations_r = orientations[1]    

        self.locations_l = locations[0]
        self.locations_r = locations[1]

        self.pose = data.pose
        self.pose_parser = PosePlotter(pose = self.pose)
        self.velocity_left = data.velocity_left
        self.velocity_right = data.velocity_right

        # Handle handshapes
        if self.base_filename is not None and handshapes is not None:
            try:
                print('\n Evaluating file: ', self.base_filename)
                self.handshapes_R_top3 = self.handshapes_top3[self.base_filename + '-R.pkl']
                self.handshapes_L_top3 = self.handshapes_top3[self.base_filename + '-L.pkl']
            except TypeError:
                print("TypeError occured")
                self.handshapes_R_top3 = self.handshapes_top3[0][self.base_filename + '-R.pkl']
                self.handshapes_L_top3 = self.handshapes_top3[0][self.base_filename + '-L.pkl']

            self.handshapes_R = self.handshapes[self.base_filename + '-R.pkl']
            print(self.handshapes_R)
            
            self.handshapes_L = self.handshapes[self.base_filename + '-L.pkl']
            #self.handshapes_R = label_smoothing_and_transitions(self.handshapes_R)
            #self.handshapes_L = label_smoothing_and_transitions(self.handshapes_L)
        else:
            self.handshapes_R = None
            self.handshapes_L = None

        # Load handpose data
        self.hamer_R = data.normalized_keypoints_hamer_right
        self.hamer_L = data.normalized_keypoints_hamer_left


        # Initialize frame storage
        self.gif_frames = []
        self.frames = data.frames
        # Directory to save frames

        self.frames_dir = os.path.join(self.base_path[:-5],'PoseTools', 'src','server', 'public', 'frames')

        if self.base_filename:
            self.frames_dir = os.path.join(self.frames_dir, self.base_filename)
        # Create frames directory if it doesn't exist
        if not os.path.exists(self.frames_dir):
            try:
                os.makedirs(self.frames_dir, exist_ok=True)
                print(f"Directory created at: {self.frames_dir}")
            except OSError as e:
                print(f"Error creating directory {self.frames_dir}: {e}")
        else:
            if not os.listdir(self.frames_dir):  # If empty
                print("The directory is empty.")
            else:  # If not empty
                # Get all .png files in the directory
                for file in os.listdir(self.frames_dir):
                    file_path = os.path.join(self.frames_dir, file)

                    # Check if it's a file (not a directory)
                    if os.path.isfile(file_path):
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            print(f"Failed to delete {file_path}: {e}")
                
                print(f"Deleted old frame files.")

        # Initialize figure and axes
        self.fig = None
        self.axes = {}
        self.scatters = {}
        self.lines = {}
        self.textbox = None


    def _determine_status(self, current, start, stop):
        # Same as before
        if start == -1:
            return 'Inactive'
        if stop == -1:
            return 'Active' if current >= start else 'Inactive'
        return 'Active' if start <= current <= stop else 'Inactive'

    def _initialize_figure(self):
        """Initialize the figure and axes for the animation."""
        self.fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(4, 5, width_ratios=[3, 2 , 2, 2, 3], figure=self.fig, wspace=0.5, hspace=0.5)

        # Video Frame
        ax_video = self.fig.add_subplot(gs[0:2, 0])
        ax_video.axis('off')
        self.axes['video'] = ax_video

        # Textbox
        ax_textbox = self.fig.add_subplot(gs[2:, :2])
        ax_textbox.axis('off')
        self.axes['textbox'] = ax_textbox
        self.textbox = ax_textbox.text(
            0.05, 0.95, '',
            ha='left', va='top',
            fontsize=12,
            wrap=True,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1.5')
        )

        # Wrist Movement Plots
        # Position Plot
        ax_plot_pos = self.fig.add_subplot(gs[2, 2:4])
        ax_plot_pos.set_title('Wrist Movement - Position')
        ax_plot_pos.set_xlabel('Frame Index')
        ax_plot_pos.set_ylabel('Normalized Movement')
        ax_plot_pos.set_xlim(0, len(self.frames))
        ax_plot_pos.set_ylim(0, 1)
        ax_plot_pos.grid(True)
        self.axes['plot_pos'] = ax_plot_pos

        # Velocity Plot
        ax_plot_vel = self.fig.add_subplot(gs[3, 2:4])
        ax_plot_vel.set_title('Wrist Movement - Velocity')
        ax_plot_vel.set_xlabel('Frame Index')
        ax_plot_vel.set_ylabel('Velocity')
        ax_plot_vel.set_xlim(0, len(self.frames))
        all_velocities = np.concatenate([self.velocity_left, self.velocity_right])
        vel_min, vel_max = np.min(all_velocities), np.max(all_velocities)
        ax_plot_vel.set_ylim(vel_min, vel_max)
        ax_plot_vel.grid(True)
        self.axes['plot_vel'] = ax_plot_vel

        # Initialize scatter plots without fixed alpha and with simplified categories
        self.scatters['pos'] = {
            'left_inactive': ax_plot_pos.scatter([], [], label='Left Inactive', color='blue', edgecolor='none'),
            'left_active': ax_plot_pos.scatter([], [], label='Left Active', color='blue', edgecolor='none'),
            'right_inactive': ax_plot_pos.scatter([], [], label='Right Inactive', color='red', edgecolor='none'),
            'right_active': ax_plot_pos.scatter([], [], label='Right Active', color='red', edgecolor='none')
        }
        ax_plot_pos.legend(loc='upper right')

        self.scatters['vel'] = {
            'left_inactive': ax_plot_vel.scatter([], [], label='Left Velocity Inactive', color='blue', edgecolor='none'),
            'left_active': ax_plot_vel.scatter([], [], label='Left Velocity Active', color='blue', edgecolor='none'),
            'right_inactive': ax_plot_vel.scatter([], [], label='Right Velocity Inactive', color='red', edgecolor='none'),
            'right_active': ax_plot_vel.scatter([], [], label='Right Velocity Active', color='red', edgecolor='none')
        }
        ax_plot_vel.legend(loc='upper right')


        # Angles for viewing the 3D plots
        self.angles = [[0, 0], [30, -30], [90, 90]] # [30, -60], 

        # Define edges connecting keypoints for plotting
        self.inward_edges = [
            [1, 0], [2, 1], [3, 2], [4, 3],     # Thumb
            [5, 0], [6, 5], [7, 6], [8, 7],     # Index Finger
            [9, 0], [10, 9], [11, 10], [12, 11],# Middle Finger
            [13, 0], [14, 13], [15, 14], [16, 15],# Ring Finger
            [17, 0], [18, 17], [19, 18], [20, 19] # Pinky Finger
        ]

        ax_skeleton = self.fig.add_subplot(gs[2, 4])  
        ax_skeleton.set_title('SMPLer-X Skeleton')
        ax_skeleton.axis('off')  # Optional: hide axes if not needed
        self.axes['skeleton'] = ax_skeleton

        ax_skeleton_side = self.fig.add_subplot(gs[3, 4]) 
        ax_skeleton_side.set_title('SMPLer-X Skeleton')
        ax_skeleton_side.axis('off')  # Optional: hide axes if not needed
        self.axes['skeleton_side'] = ax_skeleton_side
        

        # Create subplots for the right hand in a 2x2 grid
        self.scatter_pose_R_list = []
        self.lines_pose_R_list = []
        for idx_angle, angle in enumerate(self.angles):
            col =  idx_angle + 1
            ax = self.fig.add_subplot(gs[0, col], projection='3d')
            ax.set_title(f'Right Hand Angle {idx_angle+1}')
            scatter = ax.scatter([], [], [], c='b', s=20)
            lines = []
            for edge in self.inward_edges:
                line, = ax.plot([], [], [], 'r-', linewidth=1)
                lines.append(line)
            self.scatter_pose_R_list.append(scatter)
            self.lines_pose_R_list.append(lines)
            ax.view_init(elev=angle[0], azim=angle[1])
            ax.set_zlim(-0.1, 0.2)
            ax.set_xlim(-0.2, 0.2)
            ax.set_ylim(-0.2, 0.2)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

        # Create subplots for the left hand in a 2x2 grid
        self.scatter_pose_L_list = []
        self.lines_pose_L_list = []
        for idx_angle, angle in enumerate(self.angles):
            col = idx_angle + 1
            ax = self.fig.add_subplot(gs[1, col], projection='3d')
            ax.set_title(f'Left Hand Angle {idx_angle+1}')
            scatter = ax.scatter([], [], [], c='g', s=20)
            lines = []
            for edge in self.inward_edges:
                line, = ax.plot([], [], [], 'r-', linewidth=1)
                lines.append(line)
            self.scatter_pose_L_list.append(scatter)
            self.lines_pose_L_list.append(lines)
            ax.view_init(elev=angle[0], azim=angle[1])
            ax.set_zlim(-0.1, 0.2)
            ax.set_xlim(-0.2, 0.2)
            ax.set_ylim(-0.2, 0.2)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

        self.data_pos = {
            'left_inactive': ([], []),
            'left_active': ([], []),
            'right_inactive': ([], []),
            'right_active': ([], [])
        }
        self.data_vel = {
            'left_inactive': ([], []),
            'left_active': ([], []),
            'right_inactive': ([], []),
            'right_active': ([], [])
        }


        # Create bar plots
        ax_bar_plot_0 = self.fig.add_subplot(gs[0, 4])
        ax_bar_plot_0.set_title('Parwise distance Right Hand')
        self.axes['bar_plot_right'] = ax_bar_plot_0

        ax_bar_plot_1 = self.fig.add_subplot(gs[1, 4])
        ax_bar_plot_1.set_title('Parwise distance Left Hand')
        self.axes['bar_plot_left'] = ax_bar_plot_1



        #self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)
        plt.ioff()

    def _update_frame(self, idx):
        """Update the plots for each frame."""
        #start = time.time()
        # Update Video Frame
        frame = self.frames[idx]
        pose = self.pose[idx]
        self.im.set_data(frame)
        #stop = time.time()
        #print(f"Time taken to update video frame: {stop - start:.2f} seconds")
        #start   = time.time()

        # Current Frame Index
        current_frame = idx

        # Add handshape information if available
        try:
            handshape_R = self.handshapes_R[idx] if self.handshapes_R is not None else 'Unknown'
            handshape_L = self.handshapes_L[idx] if self.handshapes_L is not None else 'Unknown'
        except IndexError:
            handshape_R = 'NA'
            handshape_L = 'NA'
            print('Warning: Misaligned handshape data')
        orientation_r = self.orientations_r[idx] if self.orientations_r is not None else 'Unknown'
        orientation_l = self.orientations_l[idx] if self.orientations_l is not None else 'Unknown'

        try:
            location_r = self.locations_r[idx] if self.locations_r is not None else 'Unknown'
            location_l = self.locations_l[idx] if self.locations_l is not None else 'Unknown'
        except IndexError:
            location_r = 'NA'
            location_l = 'NA'
            print('Warning: Misaligned location data')

        try:
            handshape_R_top3 = self.handshapes_R_top3[idx] if self.handshapes_R_top3 is not None else 'Unknown'
        except IndexError: 
            handshape_R_top3 = 'NA'
        try:
            handshape_L_top3 = self.handshapes_L_top3[idx] if self.handshapes_L_top3 is not None else 'Unknown'
        except IndexError:
            handshape_L_top3 = 'NA'
        # Add handedness information if available
        try:
            handedness = self.handedness[idx] if self.handedness is not None and idx < len(self.handedness) else 'Unknown'
        except:
            handedness = 'NA'    
        
        #stop = time.time()
        #print(f"Time taken to update text: {stop - start:.2f} seconds")
        # Update Textbox Conte  nt
        #start = time.time()
        textbox_content = (
            f"{'Frame number = '}{idx}\n"
            f"{'Active Hands':<80}\n"
            f"{'':<15} {'R:':<2} {'Active' if self.right_wrist_activity[idx] else 'Inactive'}\n"
            f"{'':<15} {'L:':<2} {'Active' if self.left_wrist_activity[idx] else 'Inactive'}\n\n"
            f"{'Handshapes:':<15}\n"
            f"{'':<15} {'R:':<9} {handshape_R}\n"
            f"{'':<15} {'Top 3:':<2} {handshape_R_top3}\n"
            f"{'':<15} {'L:':<9} {handshape_L}\n"
            f"{'':<15} {'Top 3:':<2} {handshape_L_top3}\n\n"
            f"{'Handedness:':<15}\n"
            f"{'':<15} {'-:':<2} {handedness}\n\n"
            f"{'Location:':<14}\n"
            f"{'':<15} {'R:':<2} {location_r}\n"
            f"{'':<15} {'L:':<2} {location_l}\n\n"
            f"{'Orientation:':<14}\n"
            f"{'':<15} {'R:':<2} {orientation_r}\n"
            f"{'':<15} {'L:':<2} {orientation_l}\n\n"
        )
        self.textbox.set_text(textbox_content)
        #stop = time.time()
        #print(f"Time taken to update textbox: {stop - start:.2f} seconds")
        #start = time.time()
        # Update Wrist Movement Plot Data with Simplified Categories
        self._update_plot_data(
            idx, current_frame,
            self.data_pos, self.scatters['pos'],
            self.data_vel, self.scatters['vel']
        )
        #stop = time.time()
        #print(f"Time taken to update wrist movement plot data: {stop - start:.2f} seconds")
        
        # Update 3D Pose Plots for Right Hand
        #start = time.time()
        try:
            current_keypoint_R = self.hamer_R[idx]
        except:
            current_keypoint_R = np.zeros((21, 3))

        x_R, y_R, z_R = current_keypoint_R[:, 0], current_keypoint_R[:, 1], current_keypoint_R[:, 2]

        # Batch update lines using Line3DCollection
        line_segments = [
            ([x_R[start], x_R[end]], [y_R[start], y_R[end]], [z_R[start], z_R[end]])
            for start, end in self.inward_edges
        ]

        for scatter, lines in zip(self.scatter_pose_R_list, self.lines_pose_R_list):
            scatter._offsets3d = (x_R, y_R, z_R)
            # Efficiently update lines
            for line, segment in zip(lines, line_segments):
                line.set_data(segment[:2])
                line.set_3d_properties(segment[2])
        #stop = time.time()
        #print(f"Time taken to plot right hand: {stop - start:.2f} seconds")
        #start = time.time()
        # Update 3D Pose Plots for Left Hand
        
        try:
            current_keypoint_L = self.hamer_L[idx]
        except:
            current_keypoint_L = np.zeros((21, 3))
        x_L, y_L, z_L = current_keypoint_L[:, 0], current_keypoint_L[:, 1], current_keypoint_L[:, 2]
        line_segments = [
            ([x_L[start], x_L[end]], [y_L[start], y_L[end]], [z_L[start], z_L[end]])
            for start, end in self.inward_edges
        ]
        
        for scatter, lines in zip(self.scatter_pose_L_list, self.lines_pose_L_list):
                scatter._offsets3d = (x_L, y_L, z_L)
                for line, segment in zip(lines, line_segments):
                    line.set_data(segment[:2])
                    line.set_3d_properties(segment[2])

        self.pose_parser.plot_mp_skeleton(
            pose,
            ax=self.axes['skeleton']
        )
        
        self.axes['skeleton'].set_title('SMPLer-X Skeleton')

        self.pose_parser.plot_mp_skeleton(
            pose,
            ax=self.axes['skeleton_side'],
            mode = 'yz'
            
        )

        # Update bar plots
        
        pdm_L = self.feature_extractor.pairwise_distance_matrix(current_keypoint_L)
        pdm_R = self.feature_extractor.pairwise_distance_matrix(current_keypoint_R)  
        
                # Define finger indices
        WRIST_BASE = 0
        THUMB = [1, 2, 3, 4]
        INDEX = [5, 6, 7, 8]
        MIDDLE = [9, 10, 11, 12]
        RING = [13, 14, 15, 16]
        PINKY = [17, 18, 19, 20]

        fingers = [THUMB, INDEX, MIDDLE, RING, PINKY]

        # Calculate distances for each category
        categories_L = []
        categories_R = []

        for finger in fingers:
            # Base-Wrist
            categories_L.append(pdm_L[finger[0], WRIST_BASE])
            categories_R.append(pdm_R[finger[0], WRIST_BASE])

            # Base-Tip
            categories_L.append(pdm_L[finger[0], finger[-1]])
            categories_R.append(pdm_R[finger[0], finger[-1]])

            # Base-Middle 2
            categories_L.append(pdm_L[finger[0], finger[2]])
            categories_R.append(pdm_R[finger[0], finger[2]])

            # Wrist-Tip
            categories_L.append(pdm_L[WRIST_BASE, finger[-1]])
            categories_R.append(pdm_R[WRIST_BASE, finger[-1]])

        # Convert distances into bar plot format
        bar_width = 0.2
        y = np.arange(len(fingers))  # Indices for fingers (THUMB, INDEX, etc.)
        categories = ['Base-Wrist', 'Base-Tip', 'Base-Middle', 'Wrist-Tip']

        # Define colors for the new palette
        colors = ['#FF6F61', '#6B5B95', '#88B04B',  '#F7CAC9']

        # Create grouped horizontal bar plots
        ax_bar_plot_0 = self.axes['bar_plot_right']
        ax_bar_plot_0.clear()
        ax_bar_plot_0.set_title('Right Hand Feature Distances', fontsize=14)
        ax_bar_plot_0.set_xlim(0, max(categories_R) * 1.1)
        ax_bar_plot_0.set_xlabel('Distance', fontsize=12)
        ax_bar_plot_0.set_yticks(y)
        ax_bar_plot_0.set_yticklabels(['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'])

        for i, category in enumerate(categories):
            ax_bar_plot_0.barh(y - 2 * bar_width + i * bar_width, categories_R[i::4], bar_width, color=colors[i], label=category)

        legend = ax_bar_plot_0.legend(
            fontsize=10,
            loc='upper left',  # Place the legend at the top-right corner of the plot
            bbox_to_anchor=(1.01, 1.0),  # Adjust placement slightly further to the right
            ncol=1,  # Keep one column so each rotated label stacks correctly
            frameon=False  # Optional: remove the legend box frame
        )

        for text in legend.get_texts():
            text.set_rotation(90)  # Rotate text vertically
            text.set_verticalalignment('bottom')  # Align text properly

        ax_bar_plot_1 = self.axes['bar_plot_left']
        ax_bar_plot_1.clear()
        ax_bar_plot_1.set_title('Left Hand Feature Distances', fontsize=14)
        ax_bar_plot_1.set_xlim(0, max(categories_L) * 1.1)
        ax_bar_plot_1.set_xlabel('Distance', fontsize=12)
        ax_bar_plot_1.set_yticks(y)
        ax_bar_plot_1.set_yticklabels(['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'])

        for i, category in enumerate(categories):
            ax_bar_plot_1.barh(y - 2 * bar_width + i * bar_width, categories_L[i::4], bar_width, color=colors[i], label=category)

        #ax_bar_plot_1.legend(fontsize=10)

        
        if self.save_frames:
            frame_filename = f"frame_{idx:04d}.png"
            
            frame_path = os.path.join(self.frames_dir, frame_filename)
            self.fig.savefig(frame_path, pad_inches=0)
        
        
        return []

    
    def _update_plot_data(self, idx, current, data_pos, scatters_pos, data_vel, scatters_vel):
        """Update plot data based on wrist activity for both position and velocity."""
        frame_idx = idx
        
        # Retrieve activity states from selected arrays
        activity_L = self.left_wrist_activity_b[idx]
        activity_R = self.right_wrist_activity_b[idx]
        
        alpha_L = 0.2 if activity_L == 0 else 1.0
        alpha_R = 0.2 if activity_R == 0 else 1.0

        # Helper function to update data and scatter
        def update_category(side, data_dict, scatter_dict, data_value, alpha):
            if side == 'left':
                category_inactive = 'left_inactive'
                category_active = 'left_active'
                base_color = (0, 0, 1)  # blue
            else:
                category_inactive = 'right_inactive'
                category_active = 'right_active'
                base_color = (1, 0, 0)  # red

            # Determine category based on alpha (inactive or active)
            if alpha == 0.2:  # Inactive
                category = category_inactive
            else:  # Active
                category = category_active
            
            # Append the current frame index and data value to the appropriate category
            data_dict[category][0].append(frame_idx)
            data_dict[category][1].append(data_value)
            
            # Update scatter plot offsets with new data
            scatter_dict[category].set_offsets(np.c_[data_dict[category][0], data_dict[category][1]])
            
            # Create RGBA colors with dynamic alpha
            num_points = len(data_dict[category][0])
            facecolors = [(*base_color, alpha) for _ in range(num_points)]
            scatter_dict[category].set_facecolors(facecolors)

        # Update position data with selected wrist positions and activities
        # Ensure that data_value is the position value
        position_L = self.wrist_left[idx]
        position_R = self.wrist_right[idx]
        update_category('left', data_pos, scatters_pos, position_L, alpha_L)
        update_category('right', data_pos, scatters_pos, position_R, alpha_R)

        # Update velocity data with selected wrist velocities and activities
        # Ensure that data_value is the velocity value
        velocity_L = self.velocity_left[idx]
        velocity_R = self.velocity_right[idx]
        update_category('left', data_vel, scatters_vel, velocity_L, alpha_L)
        update_category('right', data_vel, scatters_vel, velocity_R, alpha_R)


    def create_animation(self, save_path, fps=10):
        """
        Create and save the GIF animation with the specified layout.

        Args:
            save_path (str): Path to save the output GIF.
            first_movement (tuple): (Left_start_frame, Right_start_frame)
            last_movement (tuple): (Left_stop_frame, Right_stop_frame)
            fps (int, optional): Frames per second for the GIF. Defaults to 10.
        """
        # Initialize figure and axes
        self._initialize_figure()
        self.im = self.axes['video'].imshow(self.frames[0])

            # Define a progress callback function
        def progress_callback(current_frame, total_frames):
            self.progress_bar.update(current_frame - self.progress_bar.n)

        # Create the animation using FuncAnimation

        self.progress_bar = tqdm(total=len(self.frames), desc="Creating animation")
        anim = FuncAnimation(
            self.fig,
            self._update_frame,
            frames=len(self.frames),
            interval=1000 / fps,
            blit=False
        )

            
        try:
            anim.save(
                save_path,
                writer = FFMpegWriter(fps=fps),
                progress_callback=progress_callback
            )
            print(f"Animation saved to {save_path}")
        except Exception as e:
            print(f"Error saving GIF: {e}")
            traceback.print_exc()
        finally:
            plt.close(self.fig)  # Close the figure to free memory
        self.progress_bar.close()


def main_visualization(data, save_anim_path=None,sign_activity_arrays = None, boolean_activity_arrays = None, handshapes=None, handedness=None, orientations = None, locations = None, fps = 5):
        if save_anim_path:
            
            animator = PhoneticAnnotationPlot(data, sign_activity_arrays, boolean_activity_arrays, handshapes = handshapes, handedness = handedness, orientations = orientations, locations = locations)
            animator.create_animation(save_path=save_anim_path, fps=fps)

