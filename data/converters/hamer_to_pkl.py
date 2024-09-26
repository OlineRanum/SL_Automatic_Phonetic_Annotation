from PoseTools.utils.datamanagment import FileConverters

input_folder = "../signbank_videos/segmented_videos/output"
output_folder = "PoseTools/data/datasets/hamer_2a/normalized"
external_dict_file = 'PoseTools/data/metadata/metadata_1_2s.json'
dict_file = 'PoseTools/data/metadata/output_2a.txt'#'PoseTools/results/handedness.txt'

fc = FileConverters()
print('Converting...')
print('files from:', input_folder)

#fc.to_pkl(input_folder, output_folder, dict_file, external_dict_file, pose_type = 'json',  multi_hands = True)

fc.to_pkl(input_folder, output_folder, dict_file, external_dict_file, pose_type = 'json',  convert2a = True)
