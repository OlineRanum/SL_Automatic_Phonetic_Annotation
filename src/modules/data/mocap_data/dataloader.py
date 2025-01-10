import numpy as np 
import pandas as pd
from normalizer import Normalizer

class DataLoader:
    def __init__(self, file_path, mode = 'hand'):
        self.file_path = file_path
        self.df = None
        if mode == 'hand':
            self.prepare_hand_data()
            self.group = 'hand'
        elif mode == 'fullpose':
            self.edges = self.prepare_full_pose_data()
            self.group = 'fullpose'

    def prepare_hand_data(self):
        # Define the markers for the hands
        self.hands = [
            'RIDX3', 'RIDX6', 'RMID0', 'RMID6', 'RPNK3', 'RPNK6',
            'RRNG6', 'RRNG3', 'RTHM6', 'RTHM3', 'ROHAND', 'RIHAND',
            'ROWR', 'RIWR'
        ]

        self.hand_edges = [
            ['ROWR', 'RIWR'],
            ['RIHAND', 'RIWR'],
            ['ROHAND', 'RIHAND'],
            ['ROHAND', 'ROWR'],
            ['ROHAND', 'RPNK3'],
            ['ROHAND', 'RRNG3'],
            ['RIHAND', 'RMID0'],
            ['RIHAND', 'RIDX3'],
            ['RIHAND', 'RTHM3'],
            ['RIDX3', 'RIDX6'],
            ['RMID0', 'RMID6'],
            ['RPNK3', 'RPNK6'],
            ['RRNG6', 'RRNG3'],
            ['RTHM6', 'RTHM3']
        ]
        return self.hand_edges

    def prepare_full_pose_data(self):
        self.edges = [
            ['ARIEL','LFHD'],
            ['ARIEL','RFHD'],
            ['ARIEL','RBHD'],
            ['ARIEL','LBHD'],
            ['RFHD', 'LFHD'],
            ['RFHD', 'RBHD'],
            ['LFHD', 'LBHD'],
            ['LBHD','RBHD'],
            ['CLAV', 'STRN'],
            ['STRN', 'RFWT'],
            ['MFWT', 'RFWT'],
            ['RBWT', 'RFWT'],
            ['MBWT', 'RBWT'],
            ['MBWT', 'LBWT'],
            ['MFWT', 'LFWT'],
            ['LBWT', 'LFWT'],
            ['STRN', 'LFWT'],
            ['CLAV', 'LFSH'],
            ['CLAV', 'RFSH'],
            ['RTHI', 'RFWT'],
            ['LTHI', 'LFWT'],
            ['C7', 'LBSH'],
            ['C7', 'RBSH'],
            ['C7', 'T10'],
            ['RBWT', 'T10'],
            ['LBWT', 'T10'],
            ['STRN', 'T10'],
            ['RBSH', 'RFSH'],
            ['LBSH', 'LFSH'],
            ['LUPA', 'LFSH'],
            ['RUPA', 'RFSH'],
            ['RUPA', 'RIWR'],
            ['RFSH', 'RIWR'],
            ['RIEL', 'RELB'],
            ['RIWR', 'RIEL'],
            ['RELB', 'RFRM'],
            ['RFRM', 'ROWR'],
            ['ROWR',   'RIHAND'],
            ['ROHAND', 'RIHAND'],
            ['ROHAND', 'ROWR'],
            ['ROHAND', 'RPNK3'],
            ['ROHAND', 'RRNG3'],
            ['RIHAND', 'RMID0'],
            ['RIHAND', 'RIDX3'],
            ['RIHAND', 'RTHM3'],
            ['RIDX3', 'RIDX6'],
            ['RMID0', 'RMID6'],
            ['RPNK3', 'RPNK6'],
            ['RRNG6', 'RRNG3'],
            ['RTHM6', 'RTHM3'],
            ['MFWT', 'RKNI'],
            ['MBWT', 'RKNI'],
            ['MFWT', 'LKNI'],
            ['MBWT', 'LKNI'],
            ['RKNE', 'RKNI'],
            ['RSHN', 'RKNI'],
            ['RANK', 'RKNI'],
            ['LKNE', 'LKNI'],
            ['LSHN', 'LKNI'],
            ['LANK', 'LKNI'],
            ['LUPA', 'LELB'],
            ['LUPA', 'LIWR'],
            ['LIWR', 'LELB'],
            ['LELB', 'LFRM'],
            ['LFRM', 'LOWR'],
            ['LBSH', 'LUPA'],
            ['LOHAND', 'LPNK3'],
            ['LOHAND', 'LRNG3'],
            ['LIHAND', 'LMID0'],
            ['LIHAND', 'LIDX3'],
            ['LIHAND', 'LTHM3'],
            ['LIDX3', 'LIDX6'],
            ['LMID0', 'LMID6'],
            ['LPNK3', 'LPNK6'],
            ['LRNG6', 'LRNG3'],
            ['LTHM6', 'LTHM3'],
            ['RANK', 'RHEL'],
            ['RHEL', 'RMT5'],
            ['RMT5', 'RTOE'],
            ['RMT5', 'RANK'],
            ['RANK', 'RMT1'],
            ['RTOE', 'RMT1'],
            ['RMT1', 'RHEL'],
            ['RANK', 'RSHN'],
            ['RSHN', 'RKNE'],
            ['RKNE', 'RTHI'],
            ['LANK', 'LHEL'],
            ['LHEL', 'LMT5'],
            ['LMT5', 'LTOE'],
            ['LMT5', 'LANK'],
            ['LANK', 'LMT1'],
            ['LTOE', 'LMT1'],
            ['LMT1', 'LHEL'],
            ['LANK', 'LSHN'],
            ['LSHN', 'LKNE'],
            ['LKNE', 'LTHI'],
            ]
        
        self.fullpose = []
        for i in self.edges:
            self.fullpose.append(i[0])
            self.fullpose.append(i[1])

        self.fullpose = list(set(self.fullpose))
        return self.edges
        
    def load_data(self, mode = 'noframe'):
        df = self.load_marker_data(self.file_path)
        if mode == 'noframe':
            new_columns = []
        else: new_columns = ['Frame']
        if self.group == 'hand':
            group = self.hands
        elif self.group == 'fullpose':
            group = self.fullpose
        for idx in group:
            new_columns.append(idx + '<T-X>')
            new_columns.append(idx + '<T-Y>')
            new_columns.append(idx + '<T-Z>')
        self.df = df[new_columns]
        return self.df
    
    def load_marker_data(self, file_path):
        # Read the CSV file while skipping the header row ("Jose3")
        df = pd.read_csv(file_path, skiprows=1)
        return df

    def get_marker_data(self, frame):
       
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Extract columns with marker data (ignoring "Frame" column)
        marker_columns = self.df.columns[1:]
        marker_names = marker_columns[::3].str.split('<').str[0]

        try:
            frame_data = self.df[self.df['Frame'] == frame]
            if frame_data.empty:
                raise ValueError(f"Frame {frame} does not exist in the dataset.")
            marker_data = frame_data.iloc[0, 1:].values.reshape(-1, 3)
        
        except:
            frame_data = self.df.iloc[frame]
            marker_data = frame_data.values.reshape(-1, 3)
        return marker_names, marker_data

    def get_marker_arr(self):
        """
        Extract marker names and marker data from the DataFrame.
        """
        # Extract marker names by splitting the column names
        marker_columns = self.df.columns
        marker_names = marker_columns[::3].str.split('<').str[0]
        marker_data = self.df.values.reshape(len(self.df), -1, 3)
        return marker_names, marker_data

    
    
    def get_keypoint(self, df, keypoint_id, mask_nans=True):
        location = []
        for i in range(len(df)):
            # Extract X, Y, Z values
            x = df[keypoint_id + '<T-X>'].values[i]
            y = df[keypoint_id + '<T-Y>'].values[i]
            z = df[keypoint_id + '<T-Z>'].values[i]

            # Append the values (including NaNs)
            location.append([x, y, z])

        # Convert to NumPy array
        location = np.array(location)

        if mask_nans:
            # Mask the NaN values
            location = np.ma.masked_invalid(location)

        return location
    
    def get_wrist(self):
        
            # Extract the wrist location data
        wrist_location = self.get_keypoint(self.df, 'ROWR', mask_nans=True)

        # Calculate the norm across the vertical axis, using masked arrays
        norms = np.sqrt(np.nansum(np.square(wrist_location), axis=1))

        # Min-max normalization of norms, respecting masked values
        min_val = norms.min()  # Minimum value in the norms (ignoring masks)
        max_val = norms.max()  # Maximum value in the norms (ignoring masks)
        normalized_norms = (norms - min_val) / (max_val - min_val)
        return normalized_norms


    def get_hand(self):
        self.prepare_hand_data()
        marker_names = self.hands

        # Extract the corresponding data columns
        hand_columns = []
        for marker in marker_names:
            hand_columns.extend([f"{marker}<T-X>", f"{marker}<T-Y>", f"{marker}<T-Z>"])

        # Convert to NumPy array and reshape
        marker_data = self.df[hand_columns].values.reshape(len(self.df), -1, 3)

        return np.array(marker_data), marker_names