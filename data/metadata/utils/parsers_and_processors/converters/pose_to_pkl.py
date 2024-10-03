from PoseTools.utils.datamanagment import FileConverters

input_folder = "../signbank_videos/"
output_folder = "PoseTools/data/datasets/mp"

fc = FileConverters()
print('Converting...')
print('files from:', input_folder)
fc.to_pkl(input_folder, output_folder, pose_type = 'pose')

