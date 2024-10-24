import pandas as pd

# Load the first text file into a DataFrame
df_txt = pd.read_csv('/home/gomer/oline/PoseTools/src/modules/handedness/euclidean_model/results_euclidean.txt', sep=',', comment='#', header=None,
                     names=['filename', 'handedness', 'gloss', 'handshape'])

# Load the second CSV file into a DataFrame
df_csv = pd.read_csv('/home/gomer/oline/PoseTools/src/modules/handedness/euclidean_model/ground_truth_handshape.csv', sep=',')

# Merge the two DataFrames on the 'filename' column, keeping only rows that exist in both
df_merged = pd.merge(df_txt, df_csv, on='filename', how='inner')
print(len(df_merged))
df_merged = df_merged[df_merged['handedness_y'] != '1']
print(len(df_merged))
# Display the merged DataFrame

# Strip whitespace from all string columns to ensure proper matching
df_merged = df_merged.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Count the percentage where 'handshape' is equal to either 'strong_hand' or 'weak_hand'
matching_count = df_merged[(df_merged['handshape'] == df_merged['strong_hand'])].shape[0]
total_count = df_merged.shape[0]
percentage = (matching_count / total_count) * 100
print(f"Total datapoints L/R hands: {total_count}")
print(f"Percentage of rows where 'handshape' matches either 'strong_hand' or 'weak_hand': {percentage:.2f}%")

matching_count_S = df_merged[(df_merged['handshape'] == df_merged['strong_hand'])].shape[0]
percentage = (matching_count_S / total_count) * 100
print(f"Percentage of rows where 'handshape' matches 'dominant hand'': {percentage:.2f}%")

matching_count_W = df_merged[(df_merged['handshape'] == df_merged['weak_hand'])].shape[0]
percentage = (matching_count_W / total_count) * 100
print(f"Percentage of rows where 'handshape' matches 'Nondominant hand'': {percentage:.2f}%")

# Optionally, save the merged DataFrame to a new CSV file
#df_merged.to_csv('/home/gomer/oline/PoseTools/src/modules/handedness/model/merged_output.csv', index=False)