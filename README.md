# PoseTools
This repository contains code for processing pose data from sign language videos. It includes functions for automatic detection of handshapes, handedness, orientation, movement and location. 

## Table of Contents
- [Installation](#installation)
- [Data Utilities](#data-utilities)
- [File Structure](#file-structure)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:

```
   git clone https://github.com/yourusername/pose-tools.git
   cd pose-tools 
```

2. Install the environmnet 
```
    pip install -r requirements.txt
```


This repository contains code to process poses, with functionalities for automatically annotating phonetic attributes of sign languages.

## Data Utilities 

### Metadata Preparations
To prepare the metadata, configure the parameters in the metadata_processor script and run:

```
python -m PoseTools.data.parsers_and_processors.metadata_processor
```

### Pose Constructors and Format Converters 
#### Hamer 3D-Handshape pose estimation
1. Activate server environment 
```
/home/gomer/hamer/run_server.py
source .hamer/bin/activate
```
2. Run Hamer Pose Estimator
```
data/parsers_and_processors/converters/vid_to_hamer.py
```

#### PoseFromat Mediapipe Full-body Pose Extraction 
To perform pose extraction with mediapipe we use the [PoseFormat library](https://github.com/sign-language-processing/pose), which can be installed with pip
``` 
pip install pose-format
```

To convert a video to pose format:

``` 
video_to_pose --format mediapipe -i example.mp4 -o example.pose
# Or if you have a directory of videos
videos_to_poses --format mediapipe --directory /path/to/videos
```
#### Pose Format Converters
To convert between various pose formats used by different models
```
#  Pkl to Pose
python -m data/parsers_and_processors/converters/pkl_to_pose

#  Pose to Pkl
python -m data/parsers_and_processors/converters/pose_to_pkl

#  Hamer/json to Pkl
python -m data/parsers_and_processors/converters/hamer_to_pkl
```



## Models
### Euclidean Distance Approach
** Evaluation Full Directory**:
```
```
** Inference Directory with GIF**:
```
python -m src/modules/handshapes/engines/ed_algorithm/inference_with_gif
```

#### How to construct reference poses from SignBank data

```
 python -m PoseTools.src.modules.handshapes.utils.build_references
```

## Handedness 
### Evaluate number of active hands
To evaluate wheter one or two hands are signing 

``` Train NHands model 
python -m PoseTools.src.models.slgcn.train_slgcn --config nhands.yaml
```

``` Evaluate NHands model 
# Evaluate test set
python -m PoseTools.src.models.slgcn.test_slgcn --config test_nhands.yaml

# Evaluate single video
python -m PoseTools.src.models.slgcn.eval_slgcn --input_path path/to/video/video.pkl
```

#### Experimental evaluation of available models 

| Model | Checkpoints   | Classes | Train/Val/Test | a1   | a2   | rr   |
|-------|---------------|---------|----------------|------|------|------|
| SLGCN | nhands_small.ckpt    | 2       | 3739/376/382     | 0.95 | 1.00 | 0.97 |
| SLGCN | nhands_large.ckpt    | 2       | 9983/2018/-     |  |  |  |

### Something

``` Evaluate number of signing hands 
python -m PoseTools.handedness.evaluate_hand_number
```

Module to evaluate wheter the right, left or both hands are signing. To run module:

``` Evaluate Handedness
python -m PoseTools.handedness.evaluate_handedness
```

The estimation is based on differences in the integrated motion of the Center-of-Mass of each individual hand. 

## Segmentation
TBI.
