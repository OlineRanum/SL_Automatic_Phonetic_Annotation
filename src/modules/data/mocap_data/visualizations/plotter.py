import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import numpy as np
import matplotlib.pyplot as plt
import imageio
from utils.dataloader import DataLoader
from utils.normalizer import Normalizer


class Plotter:
    def __init__(self, edges):
        self.edges = edges

    def plot_3d_markers(self, marker_names, marker_data, angle_elev=30, angle_azim=30, save_path=None, group = 'hands', translate = False, label = False):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        x, y, z = marker_data[:, 0], marker_data[:, 1], marker_data[:, 2]
        if translate:
            riwr_idx = list(marker_names).index("RIWR")
            # Apply the SAME normalization transform from the reference frame
            x -= x[riwr_idx]
            y -= y[riwr_idx]
            z -= z[riwr_idx]


        for i, name in enumerate(marker_names):
            ax.scatter(x[i], y[i], z[i], c='b', marker='o')
            if label:
                ax.text(x[i], y[i], z[i], name, fontsize=9)

        name_to_index = {name: idx for idx, name in enumerate(marker_names)}
        for edge in self.edges:
            if edge[0] in name_to_index and edge[1] in name_to_index:
                idx1, idx2 = name_to_index[edge[0]], name_to_index[edge[1]]
                ax.plot([x[idx1], x[idx2]], [y[idx1], y[idx2]], [z[idx1], z[idx2]], c='k')

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('3D Marker Positions')
        ax.view_init(elev=angle_elev, azim=angle_azim)
        if group == 'hands':
            ax.set_xlim(-1, 5)
            ax.set_ylim(-2, 1)
            ax.set_zlim(0, 1)
        elif group == 'fullpose':
            ax.set_zlim(0, 1)
            ax.set_ylim(-0.3, 0.3)
            ax.set_xlim(-0.5, 0.5)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close(fig)

    def create_gif_from_frames(self, df, output_path, start_frame, end_frame, step, translation, scale_factor, rotation_matrix):
        angles = [(30, 30), (30, 0), (0, 30)]
        frames = []
        name_to_index = None

        loader = None  # We'll create a loader if needed
        # Actually we already have df, so we can reuse the DataLoader logic here:
        loader = DataLoader(None)
        loader.df = df  # We've already loaded df outside
        normalizer = Normalizer()

        for frame in range(start_frame, end_frame + 1, step):
            try:
                marker_names, marker_data = loader.get_marker_data(frame)
                normalized_data = normalizer.apply_normalization(marker_data, marker_names, translation, scale_factor, rotation_matrix)
                name_to_index = {name: idx for idx, name in enumerate(marker_names)}

                fig = plt.figure(figsize=(18, 6))
                x, y, z = normalized_data[:, 0], normalized_data[:, 1], normalized_data[:, 2]

                for i, (elev, azim) in enumerate(angles, start=1):
                    ax = fig.add_subplot(1, 3, i, projection='3d')

                    # Plot markers
                    for idx, name in enumerate(marker_names):
                        ax.scatter(x[idx], y[idx], z[idx], c='b', marker='o')
                        ax.text(x[idx], y[idx], z[idx], name, fontsize=8)

                    # Plot edges
                    for edge in self.edges:
                        if edge[0] in name_to_index and edge[1] in name_to_index:
                            idx1, idx2 = name_to_index[edge[0]], name_to_index[edge[1]]
                            ax.plot([x[idx1], x[idx2]], [y[idx1], y[idx2]], [z[idx1], z[idx2]], c='k')

                    ax.set_xlim(-1, 3)
                    ax.set_ylim(-2, 1)
                    ax.set_zlim(-2, 1)
                    ax.set_xlabel('X-axis')
                    ax.set_ylabel('Y-axis')
                    ax.set_zlabel('Z-axis')
                    ax.set_title(f"Frame {frame} - View {i}")
                    ax.view_init(elev=elev, azim=azim)

                plt.tight_layout()
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(image)

                plt.close(fig)
            except ValueError as e:
                print(f"Skipping frame {frame}: {e}")

        if frames:
            imageio.mimsave(output_path, frames, fps=5)
            print(f"GIF saved to {output_path}")
        else:
            print("No frames to include in the GIF.")