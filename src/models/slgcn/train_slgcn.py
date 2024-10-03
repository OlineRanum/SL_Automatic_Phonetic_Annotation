import omegaconf
from PoseTools.src.models.slgcn.openhands.apis.classification_model import ClassificationModel
from PoseTools.src.models.slgcn.openhands.core.exp_utils import get_trainer
import argparse
import os

# Set up argument parser
parser = argparse.ArgumentParser(description="Handedness Model Training")
parser.add_argument(
    '--config', 
    type=str, 
    default='handedness.yaml',  # Set handedness.yaml as the default
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

# Set up the trainer and model using the configuration
trainer = get_trainer(cfg)
model = ClassificationModel(cfg=cfg, trainer=trainer)

# Initialize model from checkpoint if available and train
model.init_from_checkpoint_if_available()
model.fit()
