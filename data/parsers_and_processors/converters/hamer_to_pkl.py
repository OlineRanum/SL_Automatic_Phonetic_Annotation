from PoseTools.data.parsers_and_processors.datamanagment import FileConverters

input_folder = "../../../../mnt/fishbowl/gomer/oline/hamer"
output_folder = "../../../../mnt/fishbowl/gomer/oline/hamer_pkl"
external_dict_file = None #'PoseTools/data/metadata/metadata_1_2s.json'
dict_file = None #'PoseTools/data/metadata/output_2a.txt'#'PoseTools/results/handedness.txt'

fc = FileConverters()
print('Converting...')
print('files from:', input_folder)

#fc.to_pkl(input_folder, output_folder, dict_file, external_dict_file, pose_type = 'json',  multi_hands = True)

fc.to_pkl(input_folder, output_folder, dict_file, external_dict_file, pose_type = 'json')
