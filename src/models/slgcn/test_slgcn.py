import sys, argparse, os
import  omegaconf
from PoseTools.src.models.slgcn.openhands.apis.inference import InferenceModel

# Set up argument parser
parser = argparse.ArgumentParser(description="Handedness Model Training")
parser.add_argument(
    '--config', 
    type=str, 
    default='handedness_test.yaml',  # Set handedness.yaml as the default
    help="Filename for handedness.yaml configuration file (default: handedness.yaml)"
)
# Parse arguments
args = parser.parse_args()

# Predefined path (existing path where config files are stored)
predefined_path = "PoseTools/src/models/slgcn/configs/"

# Concatenate the predefined path with the file provided via the command line
full_config_path = os.path.join(predefined_path, args.config)

# Load the configuration file
cfg = omegaconf.OmegaConf.load(full_config_path)

model = InferenceModel(cfg=cfg)
model.init_from_checkpoint_if_available()
if cfg.data.test_pipeline.dataset.inference_mode:
    model.test_inference()
else:
    model.compute_test_accuracy()
