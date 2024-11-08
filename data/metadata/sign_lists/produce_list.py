import json
import os
import csv

# Configuration
JSON_FILE_PATH = '/home/gomer/oline/PoseTools/data/metadata/glosses_meta.json'        # Replace with your JSON file path
OUTPUT_TXT_PATH = '/home/gomer/oline/PoseTools/data/metadata/sign_lists/SB_list.txt'      # Replace with your desired output TXT file path
PKL_DIRECTORY = '../../../../mnt/fishbowl/gomer/oline/hamer_pkl'     # Replace with the directory containing .pkl files

# Allowed labels for SHS and WHS
ALLOWED_LABELS = [
    'C_spread', '5', 'S', 'B', 'C', 'Horns', 'Money', '1',
    '1_curved', 'K', 'V_curved', 'T', '5_claw', 'A', 'Baby_beak',
    'Beak', 'M', 'L2', 'Baby_C', '5m_closed', 'D', '4', 'O',
    'Beak_open', 'I', 'Y', 'N', '5r', 'L', 'V', 'W', '-1'
]

def load_json(json_path):
    """Load JSON data from a file."""
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def check_pkl_exists(gloss, pkl_dir):
    """Check if the corresponding .pkl file exists for a given gloss."""
    pkl_filename = f"{gloss}-R.pkl"
    pkl_path = os.path.join(pkl_dir, pkl_filename)
    
    return os.path.isfile(pkl_path)

def process_entries(data, pkl_dir):
    """Process JSON entries and filter based on criteria."""
    processed = []
    for entry in data:
        # Each entry is a dict with one key
        for gloss_id, attributes in entry.items():
            gloss = attributes.get("Annotation ID Gloss: Dutch", "-1").strip()
            shs = attributes.get("Strong Hand", "-1").strip()
            whs = attributes.get("Weak Hand", "-1").strip() if "Weak Hand" in attributes else "-1"
            handedness = attributes.get("Handedness", "-1").strip()
            handshape_change = attributes.get("Handshape Change", "-1").strip() if "Handshape Change" in attributes else "-1"

            if handedness != "2s":
                continue
            # Check if SHS and WHS are in allowed labels
            if shs == 'B_bent' or shs == 'B_curved': shs = 'B'
            if whs == 'B_bent' or whs == 'B_curved': whs = 'B'

            if shs not in ALLOWED_LABELS or whs not in ALLOWED_LABELS:
                continue

            # Check if the corresponding .pkl file exists
            if not check_pkl_exists(gloss, pkl_dir):
                continue
            # Append the processed entry
            processed.append({
                "Gloss": gloss,
                "SHS": shs,
                "WHS": whs,
                "Handedness": handedness,
                "Handshape-Change": handshape_change
            })

    return processed

def write_to_txt(processed_data, output_path):
    """Write the processed data to a TXT file with comma-separated values."""
    with open(output_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["Gloss", "SHS", "WHS", "Handedness", "Handshape-Change"])
        # Write each row
        for entry in processed_data:
            writer.writerow([
                entry["Gloss"],
                entry["SHS"],
                entry["WHS"],
                entry["Handedness"],
                entry["Handshape-Change"]
            ])

def main():
    # Load JSON data
    try:
        data = load_json(JSON_FILE_PATH)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    # Process entries
    processed_data = process_entries(data, PKL_DIRECTORY)
    

    # Write to output TXT file
    try:
        write_to_txt(processed_data, OUTPUT_TXT_PATH)
        print(f"Successfully wrote {len(processed_data)} entries to {OUTPUT_TXT_PATH}")
    except Exception as e:
        print(f"Error writing to TXT file: {e}")

if __name__ == "__main__":
    main()
