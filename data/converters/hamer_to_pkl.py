from PoseTools.utils.datamanagment import FileConverters

input_folder = "../signbank_videos/segmented_videos/"
output_folder = "PoseTools/data/datasets/hammer_1h/segmented"

fc = FileConverters()
print('Converting...')
print('files from:', input_folder)
fc.to_pkl(input_folder, output_folder, pose_type = 'hamer')
