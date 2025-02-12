
import json
from src.models.euclidean_model

def load_references(file_path):
    # TODO: Load json file 
    # {V: [[], [], []]
    #  S: [[], [], []]}

    with open(file_path, "r", encoding="utf-8") as f:
        data_dict = json.load(f)

    return data_dict






def main():
    # TODO: Load reference handshape json 
    file_path = "..."
    data_dict = load_references(file_path)

