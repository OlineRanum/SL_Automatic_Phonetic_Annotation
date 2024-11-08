from PoseTools.data.parsers_and_processors.datamanagment import FileConverters
import os

def main(input_folder, output_folder, delete_pose = False):
    fc = FileConverters()
    print('Converting...')
    print('files from:', input_folder)
    fc.to_pkl(input_folder, output_folder, pose_type = 'pose')

    if delete_pose:
        print('Deleting original .pose files...')
        for file in os.listdir(input_folder):
            if file.endswith('.pose'):  # Only delete files with .pose extension
                os.remove(os.path.join(input_folder, file))

