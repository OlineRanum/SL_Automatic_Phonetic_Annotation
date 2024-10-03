from PoseTools.utils.datamanagment import FileConverters

input_folder = "PoseTools/data/datasets/mp"
output_folder = "PoseTools/data/datasets/mp_pose"

fc = FileConverters()
print('Converting...')
print('files from:', input_folder)
fc.to_pose(input_folder, output_folder, pose_type = 'pkl', multi_hands = True)
