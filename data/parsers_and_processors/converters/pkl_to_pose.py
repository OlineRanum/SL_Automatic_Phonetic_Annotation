from PoseTools.data.parsers_and_processors.datamanagment import FileConverters

input_folder = "/mnt/fishbowl/gomer/oline/hamer_pkl"
output_folder = "/mnt/fishbowl/gomer/oline/hamer_pose"

fc = FileConverters()
print('Converting...')
print('files from:', input_folder)
fc.hamer_to_pose(input_folder, output_folder, pose_type = 'pkl', multi_hands = True)
