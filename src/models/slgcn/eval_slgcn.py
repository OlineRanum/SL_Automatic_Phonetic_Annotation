import sys
import argparse
import os
import torch
import omegaconf
from PoseTools.src.models.slgcn.openhands.apis.inference import InferenceModel
from PoseTools.data.parsers_and_processors.parsers import PklParser
from PoseTools.src.utils.preprocessing import CenterAndScaleNormalize

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Handedness Model Evaluation")
    parser.add_argument(
        '--config', 
        type=str, 
        default='handedness_test.yaml',  # Default to handedness.yaml configuration file
        help="Path to configuration file (default: handedness_test.yaml)"
    )
    parser.add_argument(
        '--input_path', 
        type=str, 
        required=True,  # This is required
        help="Path to the input .pkl file"
    )
    args = parser.parse_args()

    # Predefined path to the configuration files
    predefined_path = "PoseTools/src/models/slgcn/configs/"
    full_config_path = os.path.join(predefined_path, args.config)

    # Load the configuration file
    cfg = omegaconf.OmegaConf.load(full_config_path)

    # Initialize the model
    model = InferenceModel(cfg=cfg)
    model.init_from_checkpoint_if_available()

    # Load data from the input_path provided as a command-line argument
    pkl_parser = PklParser()
    pose, conf = pkl_parser.read_pkl(format='normal', input_path=args.input_path)

    # Preprocess the pose data (select only 2D coordinates)
    pose = pose[:, :, :2]
    preprocessor = CenterAndScaleNormalize("shoulder_mediapipe_holistic_minimal_27")

    # Convert to PyTorch tensor and preprocess
    pose = torch.from_numpy(pose).float()
    pose = preprocessor(pose.permute(2, 0, 1))  # Rearrange (T, V, C) to (C, T, V)

    # Add batch dimension
    pose = pose.unsqueeze(0)  # Convert to shape (N=1, C, T, V)

    # Apply model
    out = model.apply_model(pose)
    logits = out[0]  # Extract logits from the model output

    # Compute probabilities using softmax
    probabilities = torch.softmax(logits, dim=1)

    # Convert to CPU and round probabilities for clarity
    rounded_probabilities = probabilities.cpu().detach().numpy().round(4)

    # Find the predicted class
    predicted_class = torch.argmax(probabilities, dim=1).item()

    # Print results
    print("Probabilities:", rounded_probabilities)
    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    main()
