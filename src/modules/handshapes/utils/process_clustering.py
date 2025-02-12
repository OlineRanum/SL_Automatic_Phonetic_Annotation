#!/usr/bin/env python
import sys
import os
import json

def main():
    # sys.argv[0] is the script name
    # sys.argv[1] -> dataType (e.g. "mocap", "video")
    # sys.argv[2] -> k (integer)
    # sys.argv[3] -> precropped ("true" or "false")
    # sys.argv[4] -> filesArg (comma-separated file names)

    if len(sys.argv) < 5:
        print(json.dumps({"error": "Insufficient arguments"}))
        sys.exit(1)

    data_type = sys.argv[1]
    k = int(sys.argv[2])
    precropped_str = sys.argv[3].lower()
    precropped = (precropped_str == 'true')
    files_arg = sys.argv[4]
    files_list = files_arg.split(',') 

    result = {
        "dataType": data_type,
        "k": k,
        "precropped": precropped,
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

            # Read the CSV with pandas
            try:
                data = pd.read_csv(file_path, low_memory=False)
                data = data.iloc[:, :-1]
                numeric_data = data.select_dtypes(include=[np.number])
                data = numeric_data.dropna()  # Remove rows with any NaN
                data = data[::1000, :]
                print(data, file=sys.stderr)
            except Exception as e:
                print(json.dumps({
                    "error": f"Failed to read CSV: {file_name}",
                    "details": str(e)
                }))
                sys.exit(1)

            # If precropped, filter rows to the indexes stored in precropped_indexes
            if precropped:
                if file_name in precropped_indexes:
                    exclude_idx = precropped_indexes[file_name]
                    data = data.loc[~data.index.isin(exclude_idx)]
            
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(numeric_data)


            centers = kmeans.cluster_centers_
            print(centers, file=sys.stderr)

            # Build an array of { clusterId, center }
            file_cluster_info = []
            for cluster_id in range(k):
                # Convert center to a Python list
                center_array = centers[cluster_id].tolist()
                file_cluster_info.append({
                    "clusterId": cluster_id,
                    "center": center_array
                })

            result["clusters"].append({
                "filename": file_name,
                "numRows": len(numeric_data),
                "labelsPreview": kmeans.labels_[:10].tolist()  # first 10 labels
            })

            clustering_outputs[file_name] = file_cluster_info

        # Write the entire dictionary of cluster centers to a new JSON file
        output_json_path = '/home/oline/SL_Automatic_Phonetic_Annotation/src/server/public/output/mocap_clusters.json'
        try:
            with open(output_json_path, 'w') as f:
                json.dump(clustering_outputs, f, indent=2)
        except Exception as e:
            print(json.dumps({
                "error": f"Failed to write clustering output to JSON file",
                "details": str(e)
            }))
            
            sys.exit(1)

    elif data_type == 'video':
        raise NotImplementedError("Video processing not yet implemented.")

    else:
        result["error"] = f"Unknown data type: {data_type}"

    # Print final JSON to stdout so Node can parse it
    print(json.dumps(result))

if __name__ == "__main__":
    main()
