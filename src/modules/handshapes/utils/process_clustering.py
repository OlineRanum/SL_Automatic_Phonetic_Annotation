#!/usr/bin/env python
import sys
import os
import json

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, root_dir)  

try:
    from src.modules.data.mocap_data.utils.dataloader import DataLoader
    from src.modules.data.mocap_data.utils.normalizer import Normalizer
    from src.modules.data.mocap_data.visualize_reference_data import Plotter
except ModuleNotFoundError as e:
    print(json.dumps({f"Module not found: {e}"}))
    
    sys.exit(1)

def main():
    # sys.argv[0] is the script name
    # sys.argv[1] -> dataType (e.g. "mocap", "video")
    # sys.argv[2] -> k (integer)
    # sys.argv[3] -> precropped ("true" or "false")
    # sys.argv[4] -> filesArg (comma-separated file names)
    # sys.argv[5] -> visualize ("true" or "false")

    if len(sys.argv) < 6:
        print(json.dumps({"error": "Insufficient arguments"}))
        sys.exit(1)

    data_type = sys.argv[1]
    k = int(sys.argv[2])
    precropped_str = sys.argv[3].lower()
    precropped = (precropped_str == 'true')
    files_arg = sys.argv[4]
    files_list = files_arg.split(',') 
    visualize_str = sys.argv[5].lower()
    visualize = (visualize_str == 'true')

    result = {
        "dataType": data_type,
        "k": k,
        "precropped": precropped,
        "visualize": visualize,
        "files": files_list,
        "clusters": [] 
    }

    clustering_outputs = {}

    if data_type == 'mocap':
        import pandas as pd
        import numpy as np
        from sklearn.cluster import KMeans

        # Where to load precropped indexes
        precropped_path = '/home/oline/SL_Automatic_Phonetic_Annotation/src/server/public/output/selected_frames.json'
        precropped_indexes = {}

        if precropped and os.path.exists(precropped_path):
            with open(precropped_path, 'r') as f:
                precropped_indexes = json.load(f)  
                # e.g. { "1_curved_markerData.csv": [0, 5, 10], ... }

        # Base path for CSV files
        mocap_basepath = '/home/oline/SL_Automatic_Phonetic_Annotation/src/server/public/data/mocap'

        for file_name in files_list:
            file_path = os.path.join(mocap_basepath, file_name)            
            loader = DataLoader(file_path, mode='hand')
                
            normalizer = Normalizer(loader)

            # Load and preprocess data
            loader.load_data()
            normalizer.load_transformations()
            right_hand, marker_names_hands = loader.get_hand()
            normalized_right_handshape = normalizer.normalize_handshape(right_hand, marker_names_hands)
            hand_edges = loader.prepare_hand_data()
            
            if precropped:
                if file_name.split('_marker')[0] in precropped_indexes:
                    exclude_idx = precropped_indexes[file_name.split('_')[0]]
                    exclude_idx = [idx for idx in exclude_idx if idx < normalized_right_handshape.shape[0]]
                    normalized_right_handshape = np.delete(normalized_right_handshape, exclude_idx, axis=0)


            valid_frames = ~np.isnan(normalized_right_handshape).any(axis=(1, 2))

            # Keep only valid frames
            cleaned_array = normalized_right_handshape[valid_frames]
            normalized_right_handshape = cleaned_array[::2]

            kmeans = KMeans(n_clusters=k, random_state=42)
            reshaped = normalized_right_handshape.reshape(normalized_right_handshape.shape[0], -1)
            kmeans.fit(reshaped)
            
            centers = kmeans.cluster_centers_
            centers = centers.reshape(k, -1, 3)

            # Build an array of { clusterId, center }
            file_cluster_info = []
            for cluster_id in range(k):
                # Convert center to a Python list
                center_array = centers[cluster_id].tolist()
                file_cluster_info.append({
                    "clusterId": cluster_id,
                    "center": center_array
                })

            clustering_outputs[file_name] = file_cluster_info
            cluster_data = {f"Cluster {i}": center for i, center in enumerate(centers)}

        # Write the entire dictionary of cluster centers to a new JSON file
        output_json_path = '/home/oline/SL_Automatic_Phonetic_Annotation/src/server/public/output/mocap_clusters.json'
        if os.path.exists(output_json_path):
            try:
                with open(output_json_path, 'r') as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}  # Reset if JSON is unreadable
        else:
            existing_data = {}  # Initialize empty dictionary if file does not exist

        existing_data.update(clustering_outputs)

        try:
            with open(output_json_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
        except Exception as e:
            print(json.dumps({
                "error": "Failed to write clustering output to JSON file",
                "details": str(e)
            }))
            sys.exit(1)

        print(visualize, flush=True, file=sys.stderr)
        if visualize:
            save_dir = '/home/oline/SL_Automatic_Phonetic_Annotation/src/server/public/graphics/mocap_reference_poses'
            plotter = Plotter(cluster_data=cluster_data, marker_names_hands= marker_names_hands ,hand_edges = hand_edges, frames_dir= save_dir)
            plotter.plot_cluster_centers(file_name)


    elif data_type == 'video':
        raise NotImplementedError("Video processing not yet implemented.")

    else:
        result["error"] = f"Unknown data type: {data_type}"


    try:
        output = {
            "status": "success",
            "message": "Clustering completed."
        }
        print(json.dumps(output))
        sys.exit(0)  # Exit cleanly
    except Exception as e:
        print(json.dumps({
            "error": "Failed to generate JSON output",
            "details": str(e)
        }))
        sys.exit(1)  # Exit with error




if __name__ == "__main__":
    main()
