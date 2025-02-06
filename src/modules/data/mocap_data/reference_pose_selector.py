import json
import numpy as np 
from dataloader import DataLoader

class DataLoaderPoseSelector(DataLoader):
    def __init__(self, file_path, mode):
        super().__init__(file_path=file_path, mode=mode)
        self.json_data = {}

    def load_json(self, json_filepath):
        """
        Load data from a JSON file into a dictionary.
        """
        try:
            with open(json_filepath, 'r') as f:
                self.json_data = json.load(f)
            print(f"JSON data successfully loaded from {json_filepath}")
        except Exception as e:
            print(f"Error loading JSON file: {e}")

    def check_file_in_dict(self):
        """
        Check if the filename (excluding path and extension) exists as a key in the dictionary.
        """
        # Extract the filename without extension
        filename = self.file_path.split('/')[-1].split('.')[0].split('_marker')[0]
        if filename in self.json_data:
            print(f"Filename '{filename}' found in dictionary")
            return self.json_data[filename]
        else:
            print(f"Filename '{filename}' not found in dictionary.")
            return None


# Example usage
if __name__ == "__main__":
    filepath = '../../../server/public/data/mocap/1_curved_markerData.csv'
    json_filepath = '../../../server/public/output/selected_frames.json'

    # Create an instance of the ExtendedDataLoader
    dataloader = DataLoaderPoseSelector(file_path=filepath, mode='hand')

    # Load the data
    dataloader.load_data()

    # Data shape is number of frames x number of markers x 3
    data, marker_names = dataloader.get_hand()

    # Load JSON data
    dataloader.load_json(json_filepath)

    # Check if the file is in the dictionary
    indexes = dataloader.check_file_in_dict()
    if indexes is not None:
        # Convert indexes to a NumPy array
        indexes = np.array([int(i) for i in indexes])

        # Subselect data
        selected_data = data[indexes]

        print(f"Original data shape: {data.shape}")
        print(f"Selected data shape: {selected_data.shape}")
        print("Subselected data:")
    else:
        print("No valid indexes found; skipping subselection.")
