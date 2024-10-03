from PoseTools.data.parsers_and_processors.datamanagment import FileConverters

input_folder = "PoseTools/data/datasets/example_data"
output_folder = "PoseTools/data/datasets/example_data"

fc = FileConverters()
print('Converting...')
print('files from:', input_folder)
fc.to_pkl(input_folder, output_folder, pose_type = 'pose')

