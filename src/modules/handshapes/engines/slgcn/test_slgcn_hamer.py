import sys 
import omegaconf
from PoseTools.src.models.slgcn.openhands.apis.inference import InferenceModel




cfg = omegaconf.OmegaConf.load("PoseTools/src/models/slgcn/configs/test_hamer.yml")
model = InferenceModel(cfg=cfg)
model.init_from_checkpoint_if_available()
if cfg.data.test_pipeline.dataset.inference_mode:
    model.test_inference()
else:
    model.compute_test_accuracy()