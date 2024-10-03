import pandas as pd
import numpy as np
import json
import os
import sys

# ----------------------------------------------
# Variables
###############################################
number_handshape_classes = 60
handedness_classes = ['1', '2s', '2a']
no_handshapechanges = False
num_test = 50  # Number of test samples per class
num_validation = 100  # Number of validation samples per class
property = 'Handedness'  # Alternative: 'Strong Hand', 'Handedness'
output_folder = 'handedness_1_2s_2a'

# ----------------------------------------------
# Paths
###############################################
json_file_path = 'PoseTools/data/metadata/json_files/glosses_meta.json'
h1_file_path = 'PoseTools/results/handedness.txt'
dict_2a_file = 'PoseTools/results/2a_handedness.txt'
package_path = os.path.abspath('../')  # Adjust this path as needed
txt_file_path = 'PoseTools/data/metadata/txt_files/metadata_test.txt'
video_ids_file = 'PoseTools/data/metadata/txt_files/corrupted.txt'
output_json = 'PoseTools/data/metadata/output/'+output_folder+'/metadata_2c.json'
value_to_id_file = 'PoseTools/data/metadata/output/'+output_folder+'/value_to_id.txt'

normalized_subdirectory = '../signbank_videos/segmented_videos/output'
segmented_subdirectory = '../signbank_videos/segmented_videos'
pkl_subdirectory = 'PoseTools/data/datasets/hamer_1_2s_2a/normalized'

split_files = {
    'test': 'PoseTools/data/metadata/output/'+output_folder+'/test.txt',
    'val': 'PoseTools/data/metadata/output/'+output_folder+'/val.txt',
    'train': 'PoseTools/data/metadata/output/'+output_folder+'/train.txt'
}

# ----------------------------------------------
# Set up package path for PoseTools
###############################################
if package_path not in sys.path:
    sys.path.append(package_path)

from PoseTools.utils.processors import TxtProcessor

# ----------------------------------------------
# Load JSON file and convert to DataFrame
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

data = load_json(json_file_path)
df = json_to_dataframe(data)

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


# Filter the DataFrame based on whether the file exists
n_df = df[df.apply(file_exists_normalized, axis=1)]
s_df = df[df.apply(file_exists_segmented, axis=1)]


print('The minimum set is n_df')
df = n_df.copy()

# Print counts of 'Handedness', 'Strong Hand', and 'Weak Hand'
print('Handedness------------------')
print(df.value_counts('Handedness'))
print('Strong hand------------------')
print(df.value_counts('Strong Hand'))
print('Weak hand------------------')
print(df.value_counts('Weak Hand'))

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

df = select_num_classes(df, num_classes=number_handshape_classes)
print("Number of classes after selection:", len(df['Strong Hand'].value_counts()))

# ----------------------------------------------
# Randomize and Reset Index
###############################################

def randomize_and_reset_index(df):
    return df.sample(frac=1, random_state=1).reset_index(drop=True)

df = randomize_and_reset_index(df)

# ----------------------------------------------
# Filter Handedness Classes
###############################################

def filter_on_handedness(df, handedness_labels):
    df = df[df['Handedness'].isin(handedness_labels)]
    return df

# Filter with a list of handedness labels
print('\n-----------------------------------------------------\n ')
df = filter_on_handedness(df, handedness_classes)
print('Total number of datapoints available after filtering on handedness:', df.shape[0])

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
# Add Handedness RL Label
###############################################
def read_txt_to_dict(file_path):
    result_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            key_value = line.strip().split(',')
            key = key_value[0].replace(' ', '')
            value = key_value[1].replace(' ', '')
            result_dict[key] = value
    return result_dict

# Read h1_dict
h1_dict = read_txt_to_dict(h1_file_path)

# Read dict_2a using TxtProcessor
txt_processor = TxtProcessor(dict_2a_file)
dict_2a = txt_processor.get_2a_dict()



def process_glosses_for_handshape(df, h1_dict, dict_2a=None):
    processed_rows = []
    for _, row in df.iterrows():
        gloss = row['Annotation ID Gloss: Dutch']
        handedness = row['Handedness']

        if handedness == '1':
            try:
                new_gloss = gloss + '-' + h1_dict[gloss]
                row['Annotation ID Gloss: Dutch'] = new_gloss
                processed_rows.append(row)
            except KeyError:
                continue

        elif handedness == '2s':
            row_r = row.copy()
            row_r['Annotation ID Gloss: Dutch'] = gloss + '-R'
            processed_rows.append(row_r)

            row_l = row.copy()
            row_l['Annotation ID Gloss: Dutch'] = gloss + '-L'
            processed_rows.append(row_l)

        if dict_2a is not None and handedness == '2a':
            row_r = row.copy()
            row_r['Annotation ID Gloss: Dutch'] = gloss + '-R'
            try:
                row_r['Strong Hand'] = dict_2a[gloss + '-R']
                processed_rows.append(row_r)
            except KeyError:
                continue

            row_l = row.copy()
            row_l['Annotation ID Gloss: Dutch'] = gloss + '-L'
            try:
                row_l['Strong Hand'] = dict_2a[gloss + '-L']
                processed_rows.append(row_l)
            except KeyError:
                continue

    return pd.DataFrame(processed_rows)

def process_glosses_for_handedness(df):
    processed_rows = []
    for _, row in df.iterrows():
        handedness = row['Handedness']

        if handedness == '1' or handedness == '2s':
            processed_rows.append(row)
            
        elif handedness == '2a':
            row_r = row.copy()
            # TODO: Remove
            processed_rows.append(row_r)
            
            
    
    return pd.DataFrame(processed_rows)

print(df[property].value_counts())
# Process glosses
if property == 'Handedness':
    df = process_glosses_for_handedness(df)
elif property == 'Strong Hand':
    df = process_glosses_for_handshape(df, h1_dict, dict_2a=dict_2a)
else:
    exit()
print(df[property].value_counts())
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

# ----------------------------------------------
# Remove Corrupted Files
###############################################

with open(video_ids_file, 'r') as file:
    corrupted_video_ids = [line.strip() for line in file.readlines()]


print('Total number of datapoints available before removing corrupted files:', df.shape[0])
df = df[~df['Annotation ID Gloss: Dutch'].isin(corrupted_video_ids)]
print('Total number of datapoints available after removing corrupted files:', df.shape[0])
df.reset_index(drop=True, inplace=True)

# ----------------------------------------------
# Map Property to Unique IDs
###############################################

unique_values = df[property].unique()
value_to_id = {value: idx for idx, value in enumerate(unique_values, start=1)}
print("Mapping of property values to IDs:", value_to_id)
df['letter_id'] = df[property].map(value_to_id)
print(df['letter_id'].value_counts())

# ----------------------------------------------
# Write Value ID Mapping to Text File
###############################################

def write_dict_to_txt(dictionary, filename):
    with open(filename, 'w') as f:
        for key, value in dictionary.items():
            f.write(f"{key}: {value}\n")

write_dict_to_txt(value_to_id, value_to_id_file)
print(f"Value to ID mapping has been written to {value_to_id_file}")



# ----------------------------------------------
# Check PKL
###############################################
def file_exists_pkl(row):
    filename = f"{row['Annotation ID Gloss: Dutch']}.pkl"
    file_path = os.path.join(pkl_subdirectory, filename)
    return os.path.isfile(file_path)

#df = df[df.apply(file_exists_pkl, axis=1)]

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
        "source": "SB",
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