import pandas as pd
import json, os


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def load_metadata(pkl_subdirectory, exclude_rad = True):
    # Define the specific columns to keep and set missing values to -1
    columns_to_keep = [
        'Annotation ID Gloss: Dutch', 'Affiliation', 
        'Handedness', 'Link', 'Strong Hand', 'Weak Hand'
    ]


    def json_to_dataframe(data, columns_to_keep):
        flattened_data = []
        for entry in data:
            for key, value in entry.items():
                # Create a dictionary for each entry with only columns_to_keep
                flat_entry = {'Gloss ID': key}
                for col in columns_to_keep:
                    # Add the value if it exists, otherwise set to -1
                    flat_entry[col] = value.get(col, -1)
                flattened_data.append(flat_entry)
        
        # Convert the list of dictionaries to a DataFrame
        return pd.DataFrame(flattened_data).reset_index(drop=True)

    # Load the JSON file and convert to a DataFrame with specified columns
    data = load_json('../glosses_meta.json')
    df = json_to_dataframe(data, columns_to_keep)
    uva_list  = [f[:-6] for f in os.listdir(pkl_subdirectory) if os.path.isfile(os.path.join(pkl_subdirectory, f))]
    df_meta = df[df['Annotation ID Gloss: Dutch'].isin(uva_list)]

    import ast 
    # Ensure 'Affiliation' values are lists
    
    df_meta['Affiliation'] = df_meta['Affiliation'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Filter out rows with empty 'Affiliation' lists
    df_meta = df_meta[df_meta['Affiliation'].astype(bool)]

    # Check the value counts for all unique affiliations
    affiliation_counts = df_meta['Affiliation'].apply(tuple).value_counts()

    # Filter for rows with Affiliation exactly ['UvA']
    if exclude_rad:
        df_meta = df_meta[df_meta['Affiliation'].apply(lambda x: x == ['UvA'])]

    columns_to_keep = ['Annotation ID Gloss: Dutch', 'Affiliation', 'Handedness', 'Link', 'Strong Hand', 'Weak Hand']

    # Select only the specified columns
    df_meta = df_meta[columns_to_keep]

    return df_meta.reset_index(drop=True)



import os
import pickle
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_pose_data(df):
    # Generate the list of filenames and corresponding gloss labels
    filenames_and_labels = [
        (os.path.join('../hamer_pkl', f"{gloss}{suffix}.pkl"), gloss)
        for gloss in df['Annotation ID Gloss: Dutch'].values
        for suffix in ['-L', '-R']
    ]

    # Remove duplicates and check for existing files
    filenames_and_labels = list(set(filenames_and_labels))
    existing_files_and_labels = [
        (filename, gloss)
        for (filename, gloss) in filenames_and_labels
        if os.path.exists(filename)
    ]

    frames = []
    gloss_labels = []

    # Function to load keypoints from a single file
    def load_keypoints(filename, gloss_label):
        try:
            with open(filename, 'rb') as file:
                data = pickle.load(file)['keypoints']
                n_frames = data.shape[0]
                start_index = n_frames // 3
                end_index = 2 * n_frames // 3    
                data_ = data[start_index:end_index]
                if data_.shape[0] == 0:
                    data_ = data  # Use all frames if the middle third is empty

                # Create a list of gloss labels corresponding to the frames
                gloss_labels_ = [gloss_label] * data_.shape[0]
                return data_, gloss_labels_
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return [], []

    # Use ThreadPoolExecutor to load files in parallel
    max_workers = min(32, os.cpu_count() + 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Start the load operations
        future_to_filename = {
            executor.submit(load_keypoints, filename, gloss): filename
            for filename, gloss in existing_files_and_labels
        }

        # Use tqdm to display progress
        for future in tqdm(as_completed(future_to_filename), total=len(future_to_filename), desc='Loading frames'):
            data_, gloss_labels_ = future.result()
            frames.extend(data_)
            gloss_labels.extend(gloss_labels_)

    return np.array(frames), gloss_labels
