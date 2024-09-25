from PoseTools.utils.datamanagment import FileConverters

input_folder = "../signbank_videos/segmented_videos/output"
output_folder = "PoseTools/data/datasets/hamer_1_2s/normalized"

fc = FileConverters()
print('Converting...')
print('files from:', input_folder)
fc.to_pkl(input_folder, output_folder, pose_type = 'json', multi_hands = True)
