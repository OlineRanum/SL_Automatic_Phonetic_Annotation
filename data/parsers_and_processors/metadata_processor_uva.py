from PoseTools.data.parsers_and_processors.processors import TxtProcessor
import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm 
import sys

# ----------------------------------------------
# Variables
###############################################


number_handshape_classes = 50
handedness_classes = ['1', '2s', '2a']
no_handshapechanges = True
preselected_handshapes =  ['C_spread', '5', 'S', 'B', 'C', 'Horns', 'Money', '1', '1_curved', 'K', 'V_curved', 'T', '5_claw', 'A', 'Baby_beak', 'Beak', 'M', 'L2', 'Baby_C', '5m_closed', 'D', '4', 'O', 'Beak_open', 'I', 'Y', 'N', '5r', 'L', 'V', 'W']
affiliation_filter = False

property = 'Strong Hand'  # Alternative: 'Strong Hand', 'Handedness'
output_folder = '31c'
output_filename = '31c_SB_all'
finegrained_handedness = False

use_extention = False
num_instances = 1000000000
num_test = 1 #int(num_instances/10/number_handshape_classes)  # Number of test samples per class
print('Number of test samples per class:', num_test)
num_validation = 1#  num_test# num_test  # Number of validation samples per class

# ----------------------------------------------
# Paths
###############################################
json_file_path = 'PoseTools/data/metadata/glosses_meta.json'
#h1_file_path = 'PoseTools/results/handedness/hamer_pkl/handedness_1.txt'
#h2s_file_path = 'PoseTools/results/handedness/hamer_pkl/handedness_2s.txt'
#h2a_file_path = 'PoseTools/results/handedness/hamer_pkl/handedness_2a.txt'

package_path = os.path.abspath('../')  # Adjust this path as needed
txt_file_path = 'PoseTools/data/metadata/txt_files/metadata_test.txt'
video_ids_file = 'PoseTools/data/metadata/corrupted.txt'
output_json = 'PoseTools/data/metadata/output/'+output_folder+'/'+output_filename+'.json'
value_to_id_file = '/home/gomer/oline/PoseTools/data/metadata/output/global_value_to_id.txt'

normalized_subdirectory = None #'../signbank_videos/segmented_videos/output'
segmented_subdirectory =None #'../signbank_videos/segmented_videos'
pkl_subdirectory = '/mnt/fishbowl/gomer/oline/hamer_pkl' # 'PoseTools/data/datasets/hamer_1_2s_2a/normalized'
pose_subdirectory = None #'/mnt/fishbowl/gomer/oline/hamer_pose' # 'PoseTools/data/datasets/hamer_1_2s_2a/normalized'

split_files = {
    'test': 'PoseTools/data/metadata/output/'+output_folder+'/test.txt',
    'val': 'PoseTools/data/metadata/output/'+output_folder+'/val.txt',
    'train': 'PoseTools/data/metadata/output/'+output_folder+'/train.txt'
}


# Check if the directory exists
if not os.path.exists('PoseTools/data/metadata/output/'+output_folder):
    # Create the directory if it does not exist
    os.makedirs('PoseTools/data/metadata/output/'+output_folder)


def read_dict_from_txt(filename):
    """
    Reads a dictionary from a .txt file where each line is in the format 'key: value'.
    
    Parameters:
    - filename: The name of the input file.
    
    Returns:
    - A dictionary with the key-value pairs from the file.
    """
    value_to_id = {}
    with open(filename, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            value_to_id[int(value)] = key  # Store with value as the key and key as the value
    return value_to_id

gloss_mapping = read_dict_from_txt(value_to_id_file)

# ----------------------------------------------
# Set up package path for PoseTools
###############################################
if package_path not in sys.path:
    sys.path.append(package_path)

# ----------------------------------------------
# Load JSON metadata file
###############################################

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def json_to_dataframe(data):
    flattened_data = []
    for entry in data:
        for key, value in entry.items():
            flat_entry = {'Gloss ID': key}
            flat_entry.update(value)
            flattened_data.append(flat_entry)
    return pd.DataFrame(flattened_data)


# Load Metadata 
data = load_json(json_file_path)
df_meta = json_to_dataframe(data)

uva_list  = [f[:-6] for f in os.listdir(pkl_subdirectory) if os.path.isfile(os.path.join(pkl_subdirectory, f))]
df_meta = df_meta[df_meta['Annotation ID Gloss: Dutch'].isin(uva_list)]

import ast 
# Ensure 'Affiliation' values are lists
df_meta['Affiliation'] = df_meta['Affiliation'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Filter out rows with empty 'Affiliation' lists
df_meta = df_meta[df_meta['Affiliation'].astype(bool)]

# Check the value counts for all unique affiliations
affiliation_counts = df_meta['Affiliation'].apply(tuple).value_counts()

# Filter for rows with Affiliation exactly ['UvA']

if affiliation_filter:
    df_meta = df_meta[df_meta['Affiliation'].apply(lambda x: x == ['UvA'])]
    print(df_meta['Affiliation'].value_counts())
    print(len(df_meta['Strong Hand'].value_counts()))

# ----------------------------------------------
# Add Handedness RL Label
###############################################
import re

def process_annotation(annotation):
    # Split the annotation into parts
    parts = annotation.split('-')

    # Check if the last part is a number using a regular expression
    if re.match(r'^\d+$', parts[-1]):
        # If the last part is a number, remove it
        annotation_id_gloss = '-'.join(parts[:-1])
        source = 'Corpus'
    else:
        # Otherwise, keep the annotation unchanged
        annotation_id_gloss = annotation
        source = 'SB'
    return annotation_id_gloss, source

def read_df_meta_to_df(df_meta):
    """
    Processes the metadata DataFrame to create a new DataFrame with specified columns,
    appending suffixes 'L' and 'R' to the 'Annotation ID Gloss: Dutch' field for each row.

    Parameters:
    - df_meta (pd.DataFrame): The metadata DataFrame containing necessary fields.

    Returns:
    - pd.DataFrame: The processed DataFrame with expanded rows and specified columns.
    """
    # Define the list of column names
    columns = [
        'Gloss ID', 'Lemma ID Gloss: Dutch', 'Annotation ID Gloss: Dutch',
        'Annotation ID Gloss: English', 'Senses: Dutch',
        'Annotation Instructions', 'Handedness', 'Strong Hand',
        'Strong Hand Letter', 'In The Web Dictionary',
        'Is This A Proposed New Sign?', 'Exclude From Ecv', 'Repeated Movement',
        'Alternating Movement', 'Link', 'Video', 'Affiliation',
        'Senses: English', 'Weak Hand', 'Relative Orientation: Movement',
        'Movement Direction', 'Tags', 'Movement Shape', 'Orientation Change',
        'Location', 'Lemma ID Gloss: English', 'Virtual Object',
        'Relative Orientation: Location', 'Strong Hand Number',
        'Handshape Change', 'Weak Hand Number', 'Phonetic Variation', 'Notes',
        'Contact Type', 'Sequential Morphology', 'Semantic Field', 'Word Class',
        'Weak Drop', 'Relation Between Articulators', 'Phonology Other',
        'Iconic Image', 'Named Entity', 'Weak Prop', 'Mouth Gesture',
        'Weak Hand Letter', 'Simultaneous Morphology', 'NME Videos',
        'Concepticon Concept Set', 'Blend Morphology', 'Mouthing',
        'LR_value', 'letter_id', 'source'
    ]

    # Initialize a list to store each row of data
    rows = []

    # Iterate over each row in df_meta
    for index, row in df_meta.iterrows():
        for suffix in ['L', 'R']:
            # Create a new row dictionary
            new_row = {
                'Handedness': row.get('Handedness', None),
                'Annotation ID Gloss: Dutch': f"{row.get('Annotation ID Gloss: Dutch', '').strip()}-{suffix}",
                'gloss': row.get('Strong Hand', None),
                'letter_id': row.get('Strong Hand', None),
                'LR_value': suffix,
                'Strong Hand': row.get('Strong Hand', None),
                'source': 'SB',  # As per your requirement
                # Copy relevant metadata fields if they exist
                'Gloss ID': row.get('Gloss ID', None),
                'Annotation ID Gloss: English': row.get('Annotation ID Gloss: English', None),
                'Lemma ID Gloss: Dutch': row.get('Lemma ID Gloss: Dutch', None),
                'Senses: Dutch': row.get('Senses: Dutch', None),
                'Annotation Instructions': row.get('Annotation Instructions', None),
                'In The Web Dictionary': row.get('In The Web Dictionary', None),
                'Is This A Proposed New Sign?': row.get('Is This A Proposed New Sign?', None),
                'Exclude From Ecv': row.get('Exclude From Ecv', None),
                'Repeated Movement': row.get('Repeated Movement', None),
                'Alternating Movement': row.get('Alternating Movement', None),
                'Link': row.get('Link', None),
                'Video': row.get('Video', None),
                'Affiliation': row.get('Affiliation', None),
                'Senses: English': row.get('Senses: English', None),
                'Weak Hand': row.get('Weak Hand', None),
                'Relative Orientation: Movement': row.get('Relative Orientation: Movement', None),
                'Movement Direction': row.get('Movement Direction', None),
                'Tags': row.get('Tags', None),
                'Movement Shape': row.get('Movement Shape', None),
                'Orientation Change': row.get('Orientation Change', None),
                'Location': row.get('Location', None),
                'Lemma ID Gloss: English': row.get('Lemma ID Gloss: English', None),
                'Virtual Object': row.get('Virtual Object', None),
                'Relative Orientation: Location': row.get('Relative Orientation: Location', None),
                'Strong Hand Number': row.get('Strong Hand Number', None),
                'Handshape Change': row.get('Handshape Change', None),
                'Weak Hand Number': row.get('Weak Hand Number', None),
                'Phonetic Variation': row.get('Phonetic Variation', None),
                'Notes': row.get('Notes', None),
                'Contact Type': row.get('Contact Type', None),
                'Sequential Morphology': row.get('Sequential Morphology', None),
                'Semantic Field': row.get('Semantic Field', None),
                'Word Class': row.get('Word Class', None),
                'Weak Drop': row.get('Weak Drop', None),
                'Relation Between Articulators': row.get('Relation Between Articulators', None),
                'Phonology Other': row.get('Phonology Other', None),
                'Iconic Image': row.get('Iconic Image', None),
                'Named Entity': row.get('Named Entity', None),
                'Weak Prop': row.get('Weak Prop', None),
                'Mouth Gesture': row.get('Mouth Gesture', None),
                'Weak Hand Letter': row.get('Weak Hand Letter', None),
                'Simultaneous Morphology': row.get('Simultaneous Morphology', None),
                'NME Videos': row.get('NME Videos', None),
                'Concepticon Concept Set': row.get('Concepticon Concept Set', None),
                'Blend Morphology': row.get('Blend Morphology', None),
                'Mouthing': row.get('Mouthing', None)
            }
            # Append the new row to the list
            rows.append(new_row)

    # Convert the list of rows into a DataFrame
    df = pd.DataFrame(rows, columns=columns)

    return df
    

# Read h1_dict
#h2a_dict = read_txt_to_df(h2a_file_path)
#h1_dict = read_txt_to_df(h1_file_path)
#h2s_dict = read_txt_to_df(h2s_file_path)
df = read_df_meta_to_df(df_meta)


def process_glosses_for_handshape(h1_dict, h2s_dict, h2a_dict):
    return  pd.concat([h1_dict, h2s_dict, h2a_dict], ignore_index=True)

def process_glosses_for_handedness(df):
    processed_rows = []
    for _, row in df.iterrows():
        handedness = row['Handedness']

        if handedness == '1':
            processed_rows.append(row)
        
        elif handedness == '2s' or handedness == '2a' or handedness == '2t':
            row_r = row.copy()
            # TODO: Remove
            if not finegrained_handedness:
                row_r['Handedness'] = '2'
            processed_rows.append(row_r)
            
            
    
    return pd.DataFrame(processed_rows)


# ----------------------------------------------
# Select Source
###############################################
#df = df[((df['source'] == 'SB'))] # | ((df['source'] == 'Corpus') & (df['Handedness'] == '2s'))] # | ((df['source'] == 'Corpus') & (df['Handedness'] == '1'))]
#labels = ['B_curved', 'Money', '1', '1_curved', 'B', '5', 'S', 'C', 'V', 'W', 'A', 'V_curved', '4', 'Baby_C', 'C_spread', 'Baby_beak_open', 'Beak', '5r', 'T', 'L', 'I', 'M', 'N', 'K', 'Y', 'Beak_open', 'B_bent', 'Beak_open_spread', '3', 'O', 'Beak_spread', 'Baby_O', 'Baby_beak', '5m']

#["B", "1", "5", "S", "C", "A", "T", "V", "C_spread", "Y", "N", "B_bent", "Money", "Baby_C", "B_curved", "1_curved", "V_curved", "L", "Beak", "5m"]

#df = df[(df['Strong Hand'].isin(labels) )]
print('Number of classes = ', len(df['Strong Hand'].value_counts()))
if preselected_handshapes is not None:
    df = df[(df['Strong Hand'].isin(preselected_handshapes))]

# ----------------------------------------------
# Remove Corrupted Files
###############################################

#with open(video_ids_file, 'r') as file:
#    corrupted_video_ids = [line.strip() for line in file.readlines()]

#print('Total number of datapoints available before removing corrupted files:', df.shape[0])
#df = df[~df['Annotation ID Gloss: Dutch'].isin(corrupted_video_ids)]
#print('Total number of datapoints available after removing corrupted files:', df.shape[0])
df.reset_index(drop=True, inplace=True)


# ----------------------------------------------
# Match with Data from Directories
###############################################

def file_exists_normalized(row):
    filename = f"normalized_{row['Annotation ID Gloss: Dutch']}_segment.json"
    file_path = os.path.join(normalized_subdirectory, filename)
    return os.path.isfile(file_path)

def file_exists_segmented(row):
    filename = f"{row['Annotation ID Gloss: Dutch']}_segment.hamer"
    file_path = os.path.join(segmented_subdirectory, filename)
    return os.path.isfile(file_path)

def file_exists_pkl(row):
    filename = f"{row['Annotation ID Gloss: Dutch']}.pkl"
    file_path = os.path.join(pkl_subdirectory, filename)
    return os.path.isfile(file_path)

def file_exists_pose(row):
    filename = f"{row['Annotation ID Gloss: Dutch']}.pose"
    file_path = os.path.join(pose_subdirectory, filename)
    return os.path.isfile(file_path)


# Filter the DataFrame based on whether the file exists
if normalized_subdirectory is not None:
    n_df = df[df.apply(file_exists_normalized, axis=1)]
    print('The minimum set is n_df')
    df = n_df.copy()
if segmented_subdirectory is not None:
    s_df = df[df.apply(file_exists_segmented, axis=1)]

if pkl_subdirectory is not None:
    df = df[df.apply(file_exists_pkl, axis=1)].reset_index(drop=True)

if pose_subdirectory is not None:
    df = df[df.apply(file_exists_pose, axis=1)].reset_index(drop=True)


# Print counts of 'Handedness', 'Strong Hand', and 'Weak Hand'
# Print counts of 'Handedness', 'Strong Hand', and 'Weak Hand'
print('\nOriginal Values ------------------------------------------------------------\n')
print(df.value_counts('Handedness').head(number_handshape_classes))
print(df.value_counts('Strong Hand').head(number_handshape_classes))
print(df.value_counts('Weak Hand').head(number_handshape_classes))
print('Total number of datapoints', df.shape[0])
print('\n-----------------------------------------------------------------------------\n')

# ----------------------------------------------
# Filter Handedness Classes
###############################################

def filter_on_handedness(df, handedness_labels):
    df = df[df['Handedness'].isin(handedness_labels)]
    print('Total number of datapoints available after filtering on handedness:', df.shape[0])
    return df

# Filter with a list of handedness labels
df = filter_on_handedness(df, handedness_classes)
print(df.value_counts('Strong Hand').head(number_handshape_classes))

# ----------------------------------------------
# Drop Handshape Changes
###############################################

def drop_handshapechanges(df):
    df = df[df['Handshape Change'].isna()]
    print(df['Handshape Change'].value_counts(dropna=False))
    return df.copy()

if no_handshapechanges:
    df = drop_handshapechanges(df)
    print('Total number of datapoints available after dropping handshape changes:', df.shape[0])

print(df.value_counts('Strong Hand').head(number_handshape_classes))
print('\n-----------------------------------------------------------------------------\n')

# ----------------------------------------------
# Select Top N Handshape Classes
###############################################

def select_num_classes(df, num_classes=35):
    strong_hand_counts = df['Strong Hand'].value_counts()
    print("\nValue counts for 'Strong Hand':")
    print(strong_hand_counts[0:num_classes])
    print("Total samples in top classes:", sum(strong_hand_counts[0:num_classes]))

    top_values = strong_hand_counts.head(num_classes).index
    return df[df['Strong Hand'].isin(top_values)].copy()


#df = select_num_classes(df, num_classes=number_handshape_classes)
#df = df[df['Strong Hand'].isin(['B', '1', 'S', 'C', 'T'])]
# Sort the DataFrame first by 'Strong Hand', then by 'Source' (prioritizing 'SB' over 'Corpus')

df = df.sort_values(by=['Strong Hand', 'source'], ascending=[True, True], key=lambda col: col.map({'SB': 0, 'Corpus': 1}))

# Group by 'Strong Hand' and take the top `num_instances` from each group
df = df.groupby('Strong Hand').head(num_instances)

print("Number of classes after selection:", len(df['Strong Hand'].value_counts()))
print(df['Strong Hand'].value_counts())
    
print('\n-----------------------------------------------------------------------------\n')

# ----------------------------------------------
# Randomize and Reset Index
###############################################

def randomize_and_reset_index(df):
    return df.sample(frac=1, random_state=1).reset_index(drop=True)

df = randomize_and_reset_index(df)

# ----------------------------------------------
# Fix Label Names
###############################################

df['Annotation ID Gloss: Dutch'] = df['Annotation ID Gloss: Dutch'].str.replace('.', '-')
df.reset_index(drop=True, inplace=True)

for gloss in df['Annotation ID Gloss: Dutch']:
    if '.' in gloss:
        print(gloss)
        exit()


# ----------------------------------------------
# Add Split to DataFrame
###############################################

def add_split(df, num_test=15, num_validation=15):
    df['split'] = np.nan
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)

    def assign_splits(df, class_label):
        class_df = df[df[property] == class_label]
        df.loc[class_df.index[:num_validation], 'split'] = 'val'
        df.loc[class_df.index[num_validation:num_validation + num_test], 'split'] = 'test'
        df.loc[class_df.index[num_test + num_validation:], 'split'] = 'train'

    for class_label in df[property].unique():
        assign_splits(df, class_label)
    return df

df = add_split(df, num_test=num_test, num_validation=num_validation)
print(df['split'].value_counts())
print('\n-----------------------------------------------------------------------------\n')

# Map the values in the dataframe column to the letter IDs from the file
#df['letter_id'] = df[property].map(value_to_id)
print(len(df['letter_id'].value_counts()))

print('\n-----------------------------------------------------------------------------\n')

# ----------------------------------------------
# Write Value ID Mapping to Text File
###############################################

def write_dict_to_txt(dictionary, filename):
    with open(filename, 'w') as f:
        for key, value in dictionary.items():
            f.write(f"{key}: {value}\n")

#write_dict_to_txt(value_to_id, value_to_id_file)

df = select_num_classes(df, num_classes=number_handshape_classes)


print(df['source'].value_counts())
print(df['split'].value_counts())
print(df['Strong Hand'].value_counts())


print('\n-----------------------------------------------------------------------------\n')

# ----------------------------------------------
# Write Metadata to JSON File
###############################################

def get_instance(row, handshape):
    return {
        "bbox": [-1, -1, -1, -1],
        "fps": 25,
        "frame_end": -1,
        "frame_start": 1,
        "instance_id": -1,
        "signer_id": -1,
        "source": row['source'],
        "split": row['split'],
        "url": "NA",
        "variation_id": -1,
        "video_id": row['Annotation ID Gloss: Dutch'],
        "camera_view": 1,
        "Minor Location": row['Relative Orientation: Location'],
        "Handshape": row['Strong Hand'],
        "Flexion": np.nan,
        "Spread": -1,
        "Sign Type": row['Handedness'],
        "Second Minor Location": np.nan,
        "Nondominant Handshape": row['Weak Hand'],
        "Sign Offset": -1,
        "Handshape Morpheme 2": np.nan,
        "Thumb Position": np.nan,
        "Major Location": row['Location'],
        "Path Movement": row['Movement Shape'],
        "Repeated Movement": row['Repeated Movement'],
        "Spread Change": -1,
        "Wrist Twist": -1,
        "Thumb Contact": -1,
        "Sign Onset": -1,
        "Contact": -1,
        "Selected Fingers": np.nan,
        "gloss": str(handshape)
    }

metadata = []
ul = 0
n_datapoints = 0
for id in df['letter_id'].unique():
    gloss_data = {
        "gloss": str(id),
        "instances": []
    }
    gloss_rows = df[df['letter_id'] == id]
    for _, row in gloss_rows.iterrows():
        instance_data = get_instance(row, id)
        gloss_data['instances'].append(instance_data)
        n_datapoints += 1
    metadata.append(gloss_data)

print('Unaccounted for Signs ', ul)
print('Number of datapoints ', n_datapoints)

with open(output_json, 'w') as json_file:
    json.dump(metadata, json_file, indent=4)

print(f"Metadata has been written to {output_json}")

# ----------------------------------------------
# Prepare Data for SignClip Format
###############################################

df['letter_id_str'] = df['letter_id'].astype(str)
df['Annotation ID Gloss: Dutch'] = '0_' + df['letter_id_str'] + '_0_' + df['Annotation ID Gloss: Dutch']
df.drop('letter_id_str', axis=1, inplace=True)

# ----------------------------------------------
# Save Filtered DataFrame to Text File
###############################################

def save_to_txt(df, file_path):
    columns_to_save = ['Gloss ID', 'Annotation ID Gloss: Dutch', 'Strong Hand', 'split']
    df_to_save = df[columns_to_save]
    with open(file_path, 'w', encoding='utf-8') as file:
        for index, row in df_to_save.iterrows():
            line = ', '.join(row.astype(str))
            file.write(line + '\n')

save_to_txt(df, txt_file_path)
print(f"\nFiltered data has been written to {txt_file_path}")

# ----------------------------------------------
# Save Split Files
###############################################

def save_split_files(df, split_files):
    for split, file_path in split_files.items():
        split_df = df[df['split'] == split]
        print(f"Saving {split} split with {len(split_df)} samples to {file_path}")
        with open(file_path, 'w', encoding='utf-8') as file:
            for lemma in split_df['Annotation ID Gloss: Dutch']:
                file.write(lemma + '\n')

save_split_files(df, split_files)

# ----------------------------------------------
# Write Summary to file
###############################################

def write_summary_to_file(file_path, df):
    with open(file_path, 'w') as file:
        file.write(f"Number of classes: {len(df['Strong Hand'].value_counts())}\n\n")
        file.write(f"Total number of datapoints: {df.shape[0]}\n\n")
        file.write(f"Number of samples per class: {num_instances}\n\n")
        file.write(f"Class distribution: {df['Strong Hand'].value_counts()}\n\n")
        file.write(f"Number of test samples per class: {num_test}\n\n")
        file.write(f"Number of validation samples per class: {num_validation}\n\n")

summary_file_path = os.path.join('PoseTools/data/metadata/output/', output_folder, output_filename + '_summary.txt')
write_summary_to_file(summary_file_path, df)