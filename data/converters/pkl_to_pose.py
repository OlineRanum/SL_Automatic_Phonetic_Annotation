from PoseTools.utils.datamanagment import FileConverters

input_folder = "PoseTools/data/datasets/hamer_1_2s/normalized"
output_folder = "PoseTools/data/datasets/hamer_1_2s/normalized/pose"

fc = FileConverters()
print('Converting...')
print('files from:', input_folder)
fc.to_pose(input_folder, output_folder, pose_type = 'pkl', multi_hands = True)
