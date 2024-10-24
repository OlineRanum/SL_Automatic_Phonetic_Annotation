# PoseTools

This repository contains code to process poses, with functionalities for automatically annotating phonetic attributes of sign languages.

## Data Utilities 

### Metadata Preparations
To prepare json files or txt files to train and evaluate models, rund module

``` Prepare metadata
python -m PoseTools.data.parsers_and_processors.metadata_processor
```


### PKL Files




## Models
### Euclidean Approach 
#### How to construct reference poses from SignBank data

```
 python -m PoseTools.src.modules.handshapes.utils.build_references
```


### Euclidean Approach 


### Euclidean Approach 
 
### Convert video to hamer
```
/home/gomer/hamer/run_server.py
source .hamer/bin/activate
```


### File Converters
As the different mode

## Additional Pose Tools
## Metadata
To prepare json files or txt files to train and evaluate models, rund module

``` Prepare metadata
python -m PoseTools.data.parsers_and_processors.metadata_processor
```

Variables and relevant paths are set on the top of the build_metadata script. 

## Pose extraction 
To perform pose extraction we use the [PoseFormat library](https://github.com/sign-language-processing/pose), which can be installed with pip
``` 
pip install pose-format
```

To convert a video to pose:
``` 
video_to_pose --format mediapipe -i example.mp4 -o example.pose
# Or if you have a directory of videos
videos_to_poses --format mediapipe --directory /path/to/videos
```

## Pose convertion 

## Segmentation
TBI.

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
