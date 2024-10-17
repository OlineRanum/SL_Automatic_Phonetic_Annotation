import json
import os
import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import random 
from PoseTools.src.modules.handedness.utils.graphics import read_dict_from_txt
class GraphDataReader:
    def __init__(self, args, fce = False):
        print('Reading data...')
        self.args = args
        self.N_NODES = 21  # Since we have 21 keypoints in hand
        self.n_spatial_edges = self.N_NODES * (self.N_NODES - 1)
            
        if fce: 
             # Build fully connected graph for a single frame
            self._build_fully_connected_edges()
    
        else:
            self.inward_edges = [
            [1, 0], [2, 1], [3, 2], [4, 3],  # Thumb
            [5, 0], [6, 5], [7, 6], [8, 7],  # Index Finger
            [9, 0], [10, 9], [11, 10], [12, 11],  # Middle Finger
            [13, 0], [14, 13], [15, 14], [16, 15],  # Ring Finger
            [17, 0], [18, 17], [19, 18], [20, 19]  # Pinky Finger
        ]
            
        
        # Load metadata
        self._load_metadata(args.root_metadata)
        
        # Load pose data from pickle files
        pickle_path = os.path.join(args.root_poses)
        data_dict = self._load_pose_data(pickle_path)

        # Build graph for each frame
        self.data_dict = self.build_graph(data_dict)

            
    
        
    def _build_fully_connected_edges(self):
        """
        Builds a fully connected edge list for a single frame.
        Every node is connected to every other node.
        """
        spatial_edges = []
        
        # Fully connect each node in the frame
        for i in range(self.N_NODES):
            for j in range(self.N_NODES):
                if i != j:  # Avoid self-loops (optional)
                    spatial_edges.append([i, j])

        self.inward_edges = torch.tensor(spatial_edges, dtype=torch.long)

    def _load_metadata(self, file_path):
        """ Load the metadata from the json file """
        with open(file_path, 'r') as file:
            metadata = json.load(file)
            
        self.gloss_dict = {}
        for item in metadata:
            self.gloss_dict.setdefault(item['gloss'], []).extend(
                [(instance['video_id'], instance['split'], instance['source'], instance['Handshape']) for instance in item['instances']])
        

    def _load_pose_data(self, pickle_path):
        """ Load the pickle files and create a dictionary with the data """
        labels = {word: index for index, word in enumerate(self.gloss_dict.keys())}
        print(labels)
        data_dict = {
            vid_id: {
                'label': labels[gloss],
                'gloss': gloss,
                'node_pos': pickle.load(open(os.path.join(pickle_path, f'{vid_id}.pkl'), 'rb'))["keypoints"][:, :, :],
                'split': split,
                'source': source,
                'Strong Hand': strong_hand
            }
            for gloss, metadata in self.gloss_dict.items()
            for vid_id, split, source, strong_hand in metadata
            if os.path.exists(os.path.join(pickle_path, f'{vid_id}.pkl'))
        }

        return data_dict

    def build_graph(self, data_dict):
        """
        Builds a graph for each frame from the pose data.
        Each graph will have 21 nodes representing hand keypoints.
        :param data_dict: A dictionary containing video data.
        :return: A dictionary containing the graph data.
        """
        graph_dict = {}
        total_vids = 0
        dropped_vids = 0
        with open('/home/gomer/oline/PoseTools/src/models/graphTransformer/data/reference_poses.pkl', 'rb') as file:
            self.reference_poses = pickle.load(file)
        self.gloss_mapping = read_dict_from_txt('/home/gomer/oline/PoseTools/data/metadata/output/global_value_to_id.txt')
    
        for vid_id, data in data_dict.items():
            num_frames = data['node_pos'].shape[0]  # Total frames in the video
            total_vids += 1
            strong_hand = data['Strong Hand']
            if (data['source'] == 'Corpus') & (num_frames > 25):
                dropped_vids += 1
                continue  # Skip videos with more than 300 frames
            for frame_idx in range(num_frames):
                # Get the node positions (keypoint coordinates) for this frame
                
                pos = torch.tensor(data['node_pos'][frame_idx, :, :], dtype=torch.float32)  # Shape [21, 3]
                closest_handshape = self.calculate_euclidean_distance(pos.numpy())
                if closest_handshape != strong_hand:
                    continue
                # Create a graph with 21 nodes (keypoints)
                edge_index = torch.tensor(self.inward_edges).t().contiguous()  # Spatial edges connecting keypoints
                
                # Add frame-level graph to the dictionary
                graph_dict[f"{vid_id}_frame_{frame_idx}"] = {
                    'label': data['label'],
                    'gloss': data['gloss'],
                    'x': pos,  # Node features (keypoint positions)
                    'edges': edge_index,  # Edge indices (spatial connections)
                    'split': data['split']
                }
        print(f"Total videos: {total_vids}, Dropped videos: {dropped_vids}")
        
        return graph_dict
    
    def calculate_euclidean_distance(self, pose):
        """
        Calculates the Euclidean distance between a pose and a reference pose for each keypoint.
        
        Parameters:
        - pose: A numpy array of shape (21, 3), representing the pose for a frame.
        - reference_pose: A numpy array of shape (21, 3), representing the reference handshape pose.
        
        Returns:
        - The Euclidean distance between the pose and the reference pose.
        """
        distances = []
        keys = []
        for key, reference_pose in self.reference_poses.items():
            distances.append(np.linalg.norm(pose - reference_pose, axis=1).mean())
            keys.append(key)
        closest_handshape = self.gloss_mapping[int(keys[np.argmin(np.array(distances))])]
        return closest_handshape
        

class GraphDataLoader:
    def __init__(self, data, args):
        print('Building dataloader...')
        self.data_dict = data.data_dict
        self.batch_size = args.batch_size
        self.args = args

        self.build_loaders()

    def build_loaders(self, single_view_mode=True):
        train_data, val_data, test_data = self._split_dataset(self.data_dict)
        self.train_loader = self._load_data(train_data)
        self.val_loader = self._load_data(val_data, shuffle=False, split='val')
        self.test_loader = self._load_data(test_data, shuffle=False, split='test')

    def _split_dataset(self, data_dict, single_view_mode=True):
        train_data = {k: v for k, v in data_dict.items() if v['split'] == 'train'}
        val_data = {k: v for k, v in data_dict.items() if v['split'] == 'val'}
        test_data = {k: v for k, v in data_dict.items() if v['split'] == 'test'}
        return train_data, val_data, test_data

    def _load_data(self, data_dict, shuffle=True, split='train'):
        data_list = []
        labels = []
        for id, data in data_dict.items():
            pos = data['x']
            edge_index = data['edges']
            label = data['label']
            labels.append(label)
            # Create PyTorch Geometric Data object
            graph_data = Data(x=pos, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))

            data_list.append(graph_data)

        print(f'Number of {split} points:', len(data_list))
        return DataLoader(data_list, batch_size=self.batch_size, shuffle=shuffle)

# Example of main script to run:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_metadata', type=str, default="subset_metadata.json", help='Metadata json file location')
    parser.add_argument('--root_poses', type=str, default="subset_selection", help='Pose data dir location')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size')
    args = parser.parse_args()

    data = GraphDataReader(args)
    pyg_loader = GraphDataLoader(data, args)
    pyg_loader.build_loaders()
