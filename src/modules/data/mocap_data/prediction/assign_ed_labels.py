import json
import numpy as np

def load_references(file_path, key_name = None):
    """Load JSON file and return as a dictionary"""
    # Check available keys in JSON
    data_dict = load_json(file_path)
    print("Available keys in JSON:", data_dict.keys())

    if key_name is not None: 
        key_name = key_name
    
    # Ensure key exists before accessing it
    if key_name not in data_dict:
        print(f"Key '{key_name}' not found in JSON.")
        return

    V = data_dict[key_name]

    # Check the data type of V
    if isinstance(V, list):
        V_arr = np.array(V)  # Convert list to NumPy array
    elif isinstance(V, dict):
        V_arr = np.array(list(V.values()))  # Convert dictionary values to NumPy array
    else:
        print(f"Unexpected data type: {type(V)}")
        return

    print("Number of elements:", len(V_arr))
    print("Array shape:", V_arr.shape)

    # Extract all 'center' values
    centers = [entry['center'] for entry in V_arr if 'center' in entry]

    # Convert centers to a NumPy array
    centers_arr = np.array(centers)

    print("Centers shape:", centers_arr.shape)
    

def load_json(file_path):
    """Load JSON file and return as a dictionary"""
    with open(file_path, "r", encoding="utf-8") as f:
        data_dict = json.load(f)
    return data_dict

def main():
    # Load reference handshapes
    file_path = "/home/oline/Documents/SL_Automatic_Phonetic_Annotation/src/server/public/output/mocap_clusters.json"
    data_dict = load_references(file_path, key_name = 'V_markerData.csv')

    # TODO Load the data 
    csv = '/home/oline/Documents/SL_Automatic_Phonetic_Annotation/src/modules/data/mocap_data/prediction/ALIEN_250212_0_MarkerData.csv'
    


if __name__ == "__main__":
    main()
