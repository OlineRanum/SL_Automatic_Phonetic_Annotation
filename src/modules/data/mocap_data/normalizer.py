
import numpy as np


class Normalizer:
    def __init__(self, loader):
        self.loader = loader

    @staticmethod
    def normalize_vector(vec):
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            return vec
        return vec / norm

    def compute_normalization_transform(self, marker_names, marker_data):
        marker_names = list(marker_names)
        try:
            riwr_idx = marker_names.index("RIWR")
            rowr_idx = marker_names.index("ROWR")
            rihand_idx = marker_names.index("RIHAND")
            rohand_idx = marker_names.index("ROHAND")
        except ValueError as e:
            raise ValueError(f"Missing key markers in marker_names: {e}")

        riwr_pos = marker_data[riwr_idx]
        rowr_pos = marker_data[rowr_idx]
        rihand_pos = marker_data[rihand_idx]
        rohand_pos = marker_data[rohand_idx]

        # Translation
        translation = riwr_pos
        translated_data = marker_data - translation

        # Scale
        u = translated_data[rowr_idx] - translated_data[riwr_idx]
        dist = np.linalg.norm(u)
        if dist < 1e-12:
            raise ValueError("RIWR and ROWR are at the same position; cannot normalize.")
        scale_factor = dist
        scaled_data = translated_data / scale_factor

        # Orientation
        u = self.normalize_vector(scaled_data[rowr_idx] - scaled_data[riwr_idx])
        hand_top_mid = 0.5 * (scaled_data[rohand_idx] + scaled_data[rihand_idx])
        v = hand_top_mid - scaled_data[riwr_idx]
        if np.linalg.norm(v) < 1e-12:
            v = scaled_data[rihand_idx] - scaled_data[riwr_idx]

        n = np.cross(u, v)
        if np.linalg.norm(n) < 1e-12:
            v = scaled_data[rihand_idx] - scaled_data[riwr_idx]
            n = np.cross(u, v)
            if np.linalg.norm(n) < 1e-12:
                n = np.array([0, 0, 1])

        n = self.normalize_vector(n)
        v_new = np.cross(n, u)
        v_new = self.normalize_vector(v_new)

        rotation_matrix = np.stack((u, v_new, n), axis=1)
        return translation, scale_factor, rotation_matrix

    def apply_normalization(self, marker_data, marker_names, translation, scale_factor, rotation_matrix):
        # The following logic (including recomputing axes) is from the original code
        riwr_idx = list(marker_names).index("RIWR")
        rowr_idx = list(marker_names).index("ROWR")
        rohand_idx = list(marker_names).index("ROHAND")

        wrist = marker_data[riwr_idx]
        leftmost = marker_data[rowr_idx]
        rightmost = marker_data[rohand_idx]

        # Compute vectors for coordinate system
        u = rightmost - leftmost
        v = wrist - leftmost

        n = np.cross(u, v)
        if np.linalg.norm(n) == 0:
            n = np.array([0, 0, 1])

        v_new = np.cross(n, u)
        u_unit = self.normalize_vector(u)
        v_unit = self.normalize_vector(v_new)
        n_unit = self.normalize_vector(n)

        rotation_matrix = np.stack((u_unit, v_unit, n_unit), axis=1)
        translated_data = (marker_data - marker_data[riwr_idx])

        u = translated_data[rowr_idx] - translated_data[riwr_idx]
        dist = np.linalg.norm(u)
        if dist < 1e-12:
            raise ValueError("RIWR and ROWR are at the same position; cannot normalize.")

        scale_factor = dist
        scaled_data = translated_data / scale_factor
        data_rotated = scaled_data @ rotation_matrix

        return data_rotated
    
    def load_transformations(self):
        reference_frame = 1

        self.marker_names, marker_data = self.loader.get_marker_data(reference_frame)
        while np.isnan(marker_data[0][0]):
            
            reference_frame += 1
            self.marker_names, marker_data = self.loader.get_marker_data(reference_frame)
        
        self.translation, self.scale_factor, self.rotation_matrix = self.compute_normalization_transform(self.marker_names, marker_data)
        normalized_ref_data = self.apply_normalization(marker_data, self.marker_names, self.translation, self.scale_factor, self.rotation_matrix)

        riwr_idx = self.marker_names.tolist().index("RIWR")
        rowr_idx = self.marker_names.tolist().index("ROWR")
        print("Reference frame RIWR:", normalized_ref_data[riwr_idx])
        print("Reference frame ROWR:", normalized_ref_data[rowr_idx])


    def normalize_handshape(self, handshape, marker_names):

        normalized_handshape = []
        for frame in handshape:
            normalized_marker_data = self.apply_normalization(frame, marker_names, self.translation, self.scale_factor, self.rotation_matrix)
            normalized_handshape.append(normalized_marker_data)
        
        return np.array(normalized_handshape)
    
    def normalize_wrist(self, wrist):
        # Calculate the norm across the vertical axis, using masked arrays
        norms = np.sqrt(np.nansum(np.square(wrist), axis=1))

        # Min-max normalization of norms, respecting masked values
        min_val = norms.min()  # Minimum value in the norms (ignoring masks)
        max_val = norms.max()  # Maximum value in the norms (ignoring masks)
        normalized_norms = (norms - min_val) / (max_val - min_val)
        return normalized_norms

    def full_pose_normalizer(self, marker_names, marker_data):
        marker_names = list(marker_names)
        ariel_idx = marker_names.index("ARIEL")
        rhel_idx = marker_names.index("RHEL")

        # Step 1: Translate so that RHEL is at the origin
        translated_data = marker_data - marker_data[rhel_idx]

        # Step 2: Determine ARIEL's new z-position
        ariel_z = translated_data[ariel_idx, 2]

        # Ensure that ARIEL is not at the same z-level as RHEL
        if abs(ariel_z) < 1e-12:
            raise ValueError("ARIEL and RHEL have zero vertical separation, cannot normalize.")

        # Step 3: Compute scale factor so that ARIEL z = 1
        scale_factor = 1.0 / ariel_z

        # Apply uniform scaling to all coordinates
        normalized_data = translated_data * scale_factor

        return normalized_data