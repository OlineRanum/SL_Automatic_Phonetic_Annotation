import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
import cv2
from tqdm import tqdm
import os, glob
import traceback
from matplotlib.animation import FuncAnimation
from PoseTools.src.modules.old_segmentation.detect_transitions import label_smoothing_and_transitions
from matplotlib.animation import PillowWriter
from PoseTools.src.modules.handedness.utils.graphics import PosePlotter
import time


def load_pkl(path):
    import pickle
    with open(path, "rb") as file:
        return pickle.load(file)['keypoints']

class WristMovementAnimator:
    def __init__(self, base_path, left_wrist, left_wrist_activity, right_wrist, right_wrist_activity, base_filename=None, skip=3, handshapes=None, handedness=None, orientations = None, locations = None, save_frames= True):
        # Initialization code
        self.video_path = os.path.join(base_path, 'video_files', f"{base_filename}.mp4") if base_filename else None
        
        self.left_wrist = left_wrist
        self.left_wrist_activity = left_wrist_activity
        self.right_wrist = right_wrist
        self.right_wrist_activity = right_wrist_activity

        self.base_filename = base_filename
        self.skip = skip
        self.save_frames = save_frames
        self.handshapes = handshapes[0]
        self.handshapes_top3 = handshapes[1]
        self.orientations_l = orientations[0]
        self.orientations_r = orientations[1]    
        self.locations_l = locations[0]
        self.locations_r = locations[1]
        self.pose_path = os.path.join(base_path, 'pose_files', f"{base_filename}.pose") if base_filename else None
        self.pose_parser = PosePlotter(self.pose_path)
        self.pose, _ = self.pose_parser.load_pose()

        # Handle handshapes
        if base_filename is not None and handshapes is not None:
            try:
                print('\n Evaluating file: ', base_filename)
                print(self.handshapes_top3.keys())
                self.handshapes_R_top3 = self.handshapes_top3[base_filename + '-R.pkl']
                self.handshapes_L_top3 = self.handshapes_top3[base_filename + '-L.pkl']
            except TypeError:
                print("TypeError occured")
                self.handshapes_R_top3 = self.handshapes_top3[0][base_filename + '-R.pkl']
                self.handshapes_L_top3 = self.handshapes_top3[0][base_filename + '-L.pkl']

            self.handshapes_R = self.handshapes[base_filename + '-R.pkl']
            self.handshapes_L = self.handshapes[base_filename + '-L.pkl']
            self.handshapes_R = label_smoothing_and_transitions(self.handshapes_R)
            self.handshapes_L = label_smoothing_and_transitions(self.handshapes_L)
        else:
            self.handshapes_R = None
            self.handshapes_L = None

        # Load handpose data
        self.hamer_R = load_pkl(os.path.join(base_path, 'hamer_pkl', f"normalized_{base_filename}-R.pkl"))
        self.hamer_L = load_pkl(os.path.join(base_path, 'hamer_pkl', f"normalized_{base_filename}-L.pkl"))
        
        # Handle handedness
        if handedness is not None:
            self.handedness = handedness[::skip]
        else:
            self.handedness = None

        # Initialize frame storage
        self.frames = []
        self.selected_frames = []
        self.selected_left_wrist = []
        self.selected_right_wrist = []
        self.gif_frames = []

        # Directory to save frames
        self.frames_dir = os.path.join(base_path[:-21], 'demo_files', 'graphics', 'gifs', 'public', 'frames')
        
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
                png_files = glob.glob(os.path.join(self.frames_dir, '*.png'))
                
                # Delete all .png files
                for file in png_files:
                    os.remove(file)
                    print(f"Deleted: {file}")

        # Initialize figure and axes
        self.fig = None
        self.axes = {}
        self.scatters = {}
        self.lines = {}
        self.textbox = None

    def get_video_frames(self):
        # Same as before
        if not self.video_path or not os.path.exists(self.video_path):
            print(f"Video file does not exist at: {self.video_path}")
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {self.video_path}")
            return

        self.frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame is None:
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frames.append(frame_rgb)
            cap.release()
            print(f"Total frames captured: {len(self.frames)}")
        except Exception as e:
            print(f"Error during frame capture: {e}")
            traceback.print_exc()
            cap.release()

    def select(self, arr, range = 5):
        arr = arr[::self.skip]
        return arr[self.start_val - range:self.stop_val + range+1]

    def set_startstop(self, range = 5):
        if abs(self.L_start - len(self.frames)) < range:
            self.start_val = self.R_start
            self.stop_val = self.R_stop
        elif abs(self.R_start - len(self.frames)) < range:	
            self.start_val = self.L_start
            self.stop_val = self.L_stop
        else:
            self.start_val = np.min([self.L_start, self.R_start])
            self.stop_val = np.max([self.L_stop, self.R_stop])


    def select_frames(self):
        # Same as before
        self.set_startstop()


        self.selected_frames = self.select(self.frames)
        self.selected_left_wrist = self.select(self.left_wrist)
        self.selected_left_wrist_activity = self.select(self.left_wrist_activity)
        self.selected_right_wrist = self.select(self.right_wrist)
        self.selected_right_wrist_activity = self.select(self.right_wrist_activity)


        try: self.handshapes_R_top3 = self.select(self.handshapes_R_top3)
        except: self.handshapes_R_top3 = None
        try: self.handshapes_L_top3 = self.select(self.handshapes_L_top3)
        except: self.handshapes_L_top3 = None

        self.selected_orientation_r = self.select(self.orientations_r)
        self.selected_orientation_l = self.select(self.orientations_l)

        self.selected_locations_r = self.select(self.locations_r)
        self.selected_locations_l = self.select(self.locations_l)

        self.handshapes_R = self.select(self.handshapes_R)
        self.handshapes_L = self.select(self.handshapes_L)

        self.selected_hamer_R = self.select(self.hamer_R)
        self.selected_hamer_L = self.select(self.hamer_L)
        self.pose = self.select(self.pose)
        
        print(f"Frames selected for GIF: {len(self.selected_frames)}")

        self.selected_velocity_left = np.diff(self.selected_left_wrist, prepend=self.selected_left_wrist[0])
        self.selected_velocity_right = np.diff(self.selected_right_wrist, prepend=self.selected_right_wrist[0])

    def _determine_status(self, current, start, stop):
        # Same as before
        if start == -1:
            return 'Inactive'
        if stop == -1:
            return 'Active' if current >= start else 'Inactive'
        return 'Active' if start <= current <= stop else 'Inactive'

    def _initialize_figure(self):
        """Initialize the figure and axes for the animation."""
        self.fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 5, width_ratios=[3, 2, 2, 2, 2], figure=self.fig, wspace=0.25, hspace=0.25)

        # Video Frame
        ax_video = self.fig.add_subplot(gs[0:2, 0])
        ax_video.axis('off')
        self.axes['video'] = ax_video

        # Textbox
        ax_textbox = self.fig.add_subplot(gs[2:, 0:2])
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
        ax_plot_pos = self.fig.add_subplot(gs[0, 3:])
        ax_plot_pos.set_title('Wrist Movement - Position')
        ax_plot_pos.set_xlabel('Frame Index')
        ax_plot_pos.set_ylabel('Normalized Movement')
        ax_plot_pos.set_xlim(0, len(self.selected_frames))
        ax_plot_pos.set_ylim(0, 1)
        ax_plot_pos.grid(True)
        self.axes['plot_pos'] = ax_plot_pos

        # Velocity Plot
        ax_plot_vel = self.fig.add_subplot(gs[1, 3:])
        ax_plot_vel.set_title('Wrist Movement - Velocity')
        ax_plot_vel.set_xlabel('Frame Index')
        ax_plot_vel.set_ylabel('Velocity')
        ax_plot_vel.set_xlim(0, len(self.selected_frames))
        all_velocities = np.concatenate([self.selected_velocity_left, self.selected_velocity_right])
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

        ax_skeleton = self.fig.add_subplot(gs[0:2, 1])  # Rows 1 to end, Column 0
        ax_skeleton.set_title('Mediapipe Skeleton')
        ax_skeleton.axis('off')  # Optional: hide axes if not needed
        self.axes['skeleton'] = ax_skeleton

        ax_skeleton_side = self.fig.add_subplot(gs[0:2, 2])  # Rows 1 to end, Column 0
        ax_skeleton_side.set_title('Mediapipe Skeleton')
        ax_skeleton_side.axis('off')  # Optional: hide axes if not needed
        self.axes['skeleton_side'] = ax_skeleton_side

        # Create subplots for the right hand in a 2x2 grid
        self.scatter_pose_R_list = []
        self.lines_pose_R_list = []
        for idx_angle, angle in enumerate(self.angles):
            col =  idx_angle + 2
            ax = self.fig.add_subplot(gs[2, col], projection='3d')
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
            col = idx_angle + 2
            ax = self.fig.add_subplot(gs[3, col], projection='3d')
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
        #self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)


    def _update_frame(self, idx):
        """Update the plots for each frame."""
        start_time = time.time()
        
        # Update Video Frame
        frame = self.selected_frames[idx]
        self.im.set_data(frame)

        # Current Frame Index
        current_frame = idx

        # Add handshape information if available
        handshape_R = self.handshapes_R[idx] if self.handshapes_R is not None else 'Unknown'
        handshape_L = self.handshapes_L[idx] if self.handshapes_L is not None else 'Unknown'

        orientation_r = self.selected_orientation_r[idx] if self.selected_orientation_r is not None else 'Unknown'
        orientation_l = self.selected_orientation_l[idx] if self.selected_orientation_l is not None else 'Unknown'

        location_r = self.selected_locations_r[idx] if self.selected_locations_r is not None else 'Unknown'
        location_l = self.selected_locations_l[idx] if self.selected_locations_l is not None else 'Unknown'

        try:
            handshape_R_top3 = self.handshapes_R_top3[idx] if self.handshapes_R_top3 is not None else 'Unknown'
        except IndexError: 
            handshape_R_top3 = 'NA'
        try:
            handshape_L_top3 = self.handshapes_L_top3[idx] if self.handshapes_L_top3 is not None else 'Unknown'
        except IndexError:
            handshape_L_top3 = 'NA'
        # Add handedness information if available
        handedness = self.handedness[idx] if self.handedness is not None and idx < len(self.handedness) else 'Unknown'

        # Update Textbox Content
        textbox_content = (
            f"{'Frame number = '}{idx}\n"
            f"{'Active Hands':<80}\n"
            f"{'':<15} {'R:':<2} {'Active' if self.selected_right_wrist_activity[idx] else 'Inactive'}\n"
            f"{'':<15} {'L:':<2} {'Active' if self.selected_left_wrist_activity[idx] else 'Inactive'}\n\n"
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
        
        # Update Wrist Movement Plot Data with Simplified Categories
        self._update_plot_data(
            idx, current_frame,
            self.data_pos, self.scatters['pos'],
            self.data_vel, self.scatters['vel']
        )
        
        # Update 3D Pose Plots for Right Hand
        if idx < len(self.selected_hamer_R):
            current_keypoint_R = self.selected_hamer_R[idx]
            x_R, y_R, z_R = current_keypoint_R[:, 0], current_keypoint_R[:, 1], current_keypoint_R[:, 2]
            for scatter, lines in zip(self.scatter_pose_R_list, self.lines_pose_R_list):
                scatter._offsets3d = (x_R, y_R, z_R)
                for line, edge in zip(lines, self.inward_edges):
                    start, end = edge
                    line.set_data([x_R[start], x_R[end]], [y_R[start], y_R[end]])
                    line.set_3d_properties([z_R[start], z_R[end]])

        # Update 3D Pose Plots for Left Hand
        if idx < len(self.selected_hamer_L):
            current_keypoint_L = self.selected_hamer_L[idx]
            x_L, y_L, z_L = current_keypoint_L[:, 0], current_keypoint_L[:, 1], current_keypoint_L[:, 2]
            for scatter, lines in zip(self.scatter_pose_L_list, self.lines_pose_L_list):
                scatter._offsets3d = (x_L, y_L, z_L)
                for line, edge in zip(lines, self.inward_edges):
                    start, end = edge
                    line.set_data([x_L[start], x_L[end]], [y_L[start], y_L[end]])
                    line.set_3d_properties([z_L[start], z_L[end]])

        # Plot the skeleton for the current frame
        self.axes['skeleton'].clear()
        self.pose_parser.plot_frame_with_edges(
            self.pose,
            frame_idx=idx,
            ax=self.axes['skeleton']
        )
        self.axes['skeleton'].set_title('Mediapipe Skeleton')
        
        self.axes['skeleton_side'].clear()
        self.pose_parser.plot_frame_with_edges(
            self.pose,
            frame_idx=idx,
            ax=self.axes['skeleton_side'],
            mode = 'yz'
        )
        

        # Save Frame if Required
        if self.save_frames:
            frame_filename = f"frame_{idx:04d}.png"
            frame_path = os.path.join(self.frames_dir, frame_filename)
            self.fig.savefig(frame_path, pad_inches=0)
        
        return []

    
    def _update_plot_data(self, idx, current, data_pos, scatters_pos, data_vel, scatters_vel):
        """Update plot data based on wrist activity for both position and velocity."""
        frame_idx = idx
        
        # Retrieve activity states from selected arrays
        activity_L = self.selected_left_wrist_activity[idx]
        activity_R = self.selected_right_wrist_activity[idx]
        
        # Determine alpha based on activity
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
        position_L = self.selected_left_wrist[idx]
        position_R = self.selected_right_wrist[idx]
        update_category('left', data_pos, scatters_pos, position_L, alpha_L)
        update_category('right', data_pos, scatters_pos, position_R, alpha_R)

        # Update velocity data with selected wrist velocities and activities
        # Ensure that data_value is the velocity value
        velocity_L = self.selected_velocity_left[idx]
        velocity_R = self.selected_velocity_right[idx]
        update_category('left', data_vel, scatters_vel, velocity_L, alpha_L)
        update_category('right', data_vel, scatters_vel, velocity_R, alpha_R)


    def create_animation(self, save_path, first_movement, last_movement, fps=10):
        """
        Create and save the GIF animation with the specified layout.

        Args:
            save_path (str): Path to save the output GIF.
            first_movement (tuple): (Left_start_frame, Right_start_frame)
            last_movement (tuple): (Left_stop_frame, Right_stop_frame)
            fps (int, optional): Frames per second for the GIF. Defaults to 10.
        """
        # Unpack movement frames
        self.L_start, self.R_start = first_movement
        
        self.L_start, self.R_start = self.L_start // self.skip, self.R_start // self.skip

        self.L_stop, self.R_stop = last_movement
        self.L_stop, self.R_stop = self.L_stop // self.skip, self.R_stop // self.skip

        # Prepare data and initialize figure
        self.get_video_frames()
        self.select_frames()
        if not self.selected_frames:
            print("No frames selected for animation.")
            return



        # Initialize figure and axes
        self._initialize_figure()
        self.im = self.axes['video'].imshow(self.selected_frames[0])

            # Define a progress callback function
        def progress_callback(current_frame, total_frames):
            self.progress_bar.update(1)

        # Create the animation using FuncAnimation
        self.progress_bar = tqdm(total=len(self.selected_frames), desc="Creating animation")
        anim = FuncAnimation(
            self.fig,
            self._update_frame,
            frames=len(self.selected_frames),
            interval=1000 / fps,
            blit=True
        )


        try:
            anim.save(
                save_path,
                writer=PillowWriter(fps=fps),
                progress_callback=progress_callback
            )
            print(f"Animation saved to {save_path}")
        except Exception as e:
            print(f"Error saving GIF: {e}")
            traceback.print_exc()
        finally:
            plt.close(self.fig)  # Close the figure to free memory
        self.progress_bar.close()
