import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend if running on a server
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm
import os
from dataloader import DataLoader
from normalizer import Normalizer
import re
class NewPlotter:
    def __init__(self, body_data, right_hand_data, right_wrist_data,
                 frames_dir, edges, hand_edges, marker_names,marker_names_hands, frames_to_skip, save_frames=True):
        """
        Args:
            body_data (ndarray):       Shape (num_frames, num_body_points, 3)
            right_hand_data (ndarray): Shape (num_frames, 21, 3) for the right hand
            right_wrist_data (ndarray):
                1D or 2D array for the 'wrist-scatter-like' plot (num_frames, ...)
            frames_dir (str):         Directory to save individual frames (optional).
            save_frames (bool):       Whether to save each frame as a .png file.
        """
        self.body_data = body_data
        self.right_hand_data = right_hand_data
        self.right_wrist_data = right_wrist_data
        self.full_pose_edges = edges
        self.hand_edges = hand_edges
        self.marker_names = marker_names
        self.marker_names_hands = marker_names_hands
        self.frames_to_skip = frames_to_skip    
        

        self.num_frames = len(body_data)
        self.frames_dir = frames_dir
        self.save_frames = save_frames

        # Create frames directory if it doesn't exist
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir, exist_ok=True)
        else:
            # Optionally clear out old frames
            for file in os.listdir(self.frames_dir):
                if file.endswith('.png'):
                    os.remove(os.path.join(self.frames_dir, file))

        # Placeholders for figure, axes, scatter objects, etc.
        self.fig = None
        self.axes = {}
        self.scatter_body = None
        self.scatter_rhand_angle1 = None
        self.scatter_rhand_angle2 = None
        self.scatter_wrist_r = None

    def _initialize_figure(self):
        """Set up the 2×3 grid layout and initialize subplots."""
        self.fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 3, width_ratios=[2, 1, 1], figure=self.fig, wspace=0.4, hspace=0.4)

        # ------------------- Row 0 -------------------
        # (0, 0) Full body 3D subplot
        ax_body_3d = self.fig.add_subplot(gs[:, 0], projection='3d')
        ax_body_3d.set_title("Full Body (Frame: 0)")
        ax_body_3d.set_xlabel('X')
        ax_body_3d.set_ylabel('Y')
        ax_body_3d.set_zlabel('Z')
        ax_body_3d.set_zlim(0, 1)
        ax_body_3d.set_ylim(-0.3, 0.3)
        ax_body_3d.set_xlim(-0.5, 0.5)
        self.axes['body'] = ax_body_3d

        # (0, 1) Right hand 3D, Angle 1
        ax_rhand_3d_1 = self.fig.add_subplot(gs[0, 1], projection='3d')
        ax_rhand_3d_1.set_title("Right Hand - Angle 1")
        ax_rhand_3d_1.set_xlabel('X')
        ax_rhand_3d_1.set_ylabel('Y')
        ax_rhand_3d_1.set_zlabel('Z')
        ax_rhand_3d_1.set_xlim(-1, 3)
        ax_rhand_3d_1.set_ylim(-2, 1)
        ax_rhand_3d_1.set_zlim(-2, 1)
        ax_rhand_3d_1.view_init(elev=30, azim=30)
        self.axes['right_hand_angle1'] = ax_rhand_3d_1

        # (0, 2) Right hand 3D, Angle 2
        ax_rhand_3d_2 = self.fig.add_subplot(gs[0, 2], projection='3d')
        ax_rhand_3d_2.set_title("Right Hand - Angle 2")
        ax_rhand_3d_2.set_xlabel('X')
        ax_rhand_3d_2.set_ylabel('Y')
        ax_rhand_3d_2.set_zlabel('Z')
        ax_rhand_3d_2.set_xlim(-1, 3)
        ax_rhand_3d_2.set_ylim(-2, 1)
        ax_rhand_3d_2.set_zlim(-2, 1)
        ax_rhand_3d_2.view_init(elev=0, azim=30)
        self.axes['right_hand_angle2'] = ax_rhand_3d_2

        # ------------------- Row 1 -------------------
        # Single wide subplot spanning all 3 columns for the wrist plot
        ax_wrist = self.fig.add_subplot(gs[1, 1:])  # Row=1, all columns
        ax_wrist.set_title("Wrist Position Plot")
        ax_wrist.set_xlabel("Frame Index")
        ax_wrist.set_ylabel("Some Value")
        ax_wrist.set_xlim(0, self.num_frames)
        ax_wrist.set_ylim(0, 1)
        self.axes['wrist'] = ax_wrist

        # Initialize empty scatter objects
        self.scatter_body = ax_body_3d.scatter([], [], [], c='blue', s=10)

        self.scatter_rhand_angle1 = ax_rhand_3d_1.scatter([], [], [], c='red', s=10)
        # For better clarity, use a different color for the second angle
        self.scatter_rhand_angle2 = ax_rhand_3d_2.scatter([], [], [], c='green', s=10)

        # 2D scatter for wrist data
        self.scatter_wrist_r = ax_wrist.scatter([], [], color='red')

    def _update_frame(self, frame_idx):
        """
        Update all plots and text for the given frame index.
        Called automatically by FuncAnimation on each frame.
        """
        # --- 1) Clear and Update Full Body 3D Plot --------------------------------------
        if frame_idx % 50 == 0:  # Update every 50 frames
            progress_percentage = int((frame_idx / self.num_frames) * 100)
            print(f"Progress: {progress_percentage}", flush=True)

        ax_body = self.axes['body']
        ax_body.cla()  # Clear the plot
        ax_body.set_title(f"Full Body (Frame: {frame_idx*self.frames_to_skip})")
        ax_body.set_xlabel('X')
        ax_body.set_ylabel('Y')
        ax_body.set_zlabel('Z')
        ax_body.set_zlim(0, 1)
        ax_body.set_ylim(-0.3, 0.3)
        ax_body.set_xlim(-0.5, 0.5)

        # Plot current body data
        current_body = self.body_data[frame_idx]  # shape (num_points, 3)
        x_b, y_b, z_b = current_body[:, 0], current_body[:, 1], current_body[:, 2]
        ax_body.scatter(x_b, y_b, z_b, c='blue', s=10)

        # Plot edges for the body
        name_to_index = {name: idx for idx, name in enumerate(self.marker_names)}
        for edge in self.full_pose_edges:
            if edge[0] in name_to_index and edge[1] in name_to_index:
                idx1, idx2 = name_to_index[edge[0]], name_to_index[edge[1]]
                ax_body.plot(
                    [x_b[idx1], x_b[idx2]],
                    [y_b[idx1], y_b[idx2]],
                    [z_b[idx1], z_b[idx2]],
                    c='k'
                )

        # --- 2) Update right hand 3D scatter: Angle 1 ------------------------
        ax_rhand_1 = self.axes['right_hand_angle1']
        ax_rhand_1.cla()  # Clear the plot
        ax_rhand_1.set_title("Right Hand - Angle 1")
        ax_rhand_1.set_xlabel('X')
        ax_rhand_1.set_ylabel('Y')
        ax_rhand_1.set_zlabel('Z')


        # Plot current hand data
        current_rhand = self.right_hand_data[frame_idx]  # shape (21, 3)
        x_r, y_r, z_r = current_rhand[:, 0], current_rhand[:, 1], current_rhand[:, 2]
        ax_rhand_1.scatter(x_r, y_r, z_r, c='red', s=10)

        # Plot edges for the right hand
        name_to_index_hands = {name: idx for idx, name in enumerate(self.marker_names_hands)}
        for edge in self.hand_edges:
            if edge[0] in name_to_index_hands and edge[1] in name_to_index_hands:
                idx1, idx2 = name_to_index_hands[edge[0]], name_to_index_hands[edge[1]]
                ax_rhand_1.plot(
                    [x_r[idx1], x_r[idx2]],
                    [y_r[idx1], y_r[idx2]],
                    [z_r[idx1], z_r[idx2]],
                    c='k'
                )

        # --- 3) Update right hand 3D scatter: Angle 2 ------------------------
        ax_rhand_2 = self.axes['right_hand_angle2']
        ax_rhand_2.cla()  # Clear the plot
        ax_rhand_2.set_title("Right Hand - Angle 2")
        ax_rhand_2.set_xlabel('X')
        ax_rhand_2.set_ylabel('Y')
        ax_rhand_2.set_zlabel('Z')


        # Plot current hand data
        ax_rhand_2.scatter(x_r, y_r, z_r, c='green', s=10)

        # Plot edges for the right hand
        for edge in self.hand_edges:
            if edge[0] in name_to_index_hands and edge[1] in name_to_index_hands:
                idx1, idx2 = name_to_index_hands[edge[0]], name_to_index_hands[edge[1]]
                ax_rhand_2.plot(
                    [x_r[idx1], x_r[idx2]],
                    [y_r[idx1], y_r[idx2]],
                    [z_r[idx1], z_r[idx2]],
                    c='k'
                )

        # --- 4) Update the wrist position scatter (2D) -----------------------
        x_vals_r = np.arange(frame_idx + 1)
        y_vals_r = self.right_wrist_data[:frame_idx + 1]
        self.scatter_wrist_r.set_offsets(np.c_[x_vals_r, y_vals_r])

        # --- 5) Optionally, save frames to disk ------------------------------
        if self.save_frames:
            frame_filename = f"frame_{frame_idx:04d}.png"
            frame_path = os.path.join(self.frames_dir, frame_filename)
            self.fig.savefig(frame_path)
        plt.close(self.fig)

        # Returning an iterable of updated artists (optional for FuncAnimation)
        
        return []


    def create_animation(self, save_path, fps=60):
        """Create and save the animation as a video (e.g., mp4) using FFMpegWriter."""
        self._initialize_figure()

        # Progress bar for feedback
        #self.progress_bar = tqdm(total=self.num_frames, desc="Creating animation")

        #def _progress_callback(current_frame, total):
        #    """ Update TQDM progress bar. """
        #    self.progress_bar.update(current_frame - self.progress_bar.n)
        #print(f"Creating animation with {self.num_frames} frames", flush=True)

        anim = FuncAnimation(
            self.fig,
            self._update_frame,
            frames=self.num_frames,
            interval=1000 / fps,
            blit=False
        )

        # Save animation
        try:
            anim.save(
                save_path,
                writer=FFMpegWriter(fps=fps)
            )
            print(f"Animation saved to {save_path}")
        except Exception as e:
            print(f"Error saving animation: {e}")
        finally:
            plt.close(self.fig)
            #self.progress_bar.close()


def process_file(file_path, frames_dir, gifs_dir, frames_to_skip=10, fps=15):
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
    name_match = re.match(r"(.*?)_markerData\.csv", filename)
    if not name_match:
        print(f"Skipping file: {file_path}. Filename format not recognized.")
        return

    name = name_match.group(1)

    # Check if the GIF already exists
    file_gif_path = os.path.join(gifs_dir, f"{name}.gif")
    if os.path.exists(file_gif_path):
        print(f"Skipping {file_path}: GIF already exists.")
        return
    else:
        print(f"Processing {file_path}...")

    # Create subdirectories for frames
    file_frames_dir = os.path.join(frames_dir, name)
    os.makedirs(file_frames_dir, exist_ok=True)

    
    loader = DataLoader(file_path, mode='fullpose')
    full_pose_edges = loader.edges
    hand_edges = loader.prepare_hand_data()
    normalizer = Normalizer(loader)

    # Load and preprocess data
    df = loader.load_data()
    marker_names, body_data = loader.get_marker_arr()
    body_pose = body_data[::frames_to_skip]
    normalized_body_data = [normalizer.full_pose_normalizer(marker_names, frame) for frame in body_pose]

    normalizer.load_transformations()
    right_hand, marker_names_hands = loader.get_hand()
    right_hand = right_hand[::frames_to_skip]
    normalized_right_handshape = normalizer.normalize_handshape(right_hand, marker_names_hands)

    right_wrist = loader.get_keypoint(df, 'ROWR', mask_nans=True)[::frames_to_skip]
    normalized_right_wrist = normalizer.normalize_wrist(right_wrist)

    # Create and save animation
    plotter = NewPlotter(
        body_data=normalized_body_data,
        right_hand_data=normalized_right_handshape,
        right_wrist_data=normalized_right_wrist,
        frames_dir=file_frames_dir,
        edges=full_pose_edges,
        hand_edges=hand_edges,
        marker_names=marker_names,
        marker_names_hands=marker_names_hands,
        frames_to_skip=frames_to_skip,
        save_frames=True
    )
    plotter.create_animation(save_path=file_gif_path, fps=fps)

def main(data_list):

    
    data_folder = '/home/oline/SL_Automatic_Phonetic_Annotation/src/server/public/data/mocap/'
    frames_root_dir = '/home/oline/SL_Automatic_Phonetic_Annotation/src/server/public/graphics/mocap_frames'
    gifs_root_dir = '/home/oline/SL_Automatic_Phonetic_Annotation/src/server/public/graphics/mocap_gifs'

    # Create root directories if they don't exist
    os.makedirs(frames_root_dir, exist_ok=True)
    os.makedirs(gifs_root_dir, exist_ok=True)

    # Process all relevant files in the data folder
    for file in data_list:
        print(f"Processing file: {file}", flush = True)
        file_path = os.path.join(data_folder, file)
        process_file(file_path, frames_root_dir, gifs_root_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process MoCap files")
    parser.add_argument('--data_list', nargs='+', help="List of data files to process", required=True)
    args = parser.parse_args()

    # Pass the `data_list` to the main function
    main(data_list=args.data_list)