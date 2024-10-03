# PoseTools

This repository contains code to process poses, with functionalities for automatically annotating phonetic attributes of sign languages


## Metadata
To prepare json files or txt files to train and evaluate models, rund module

``` Prepare metadata
python -m PoseTools.data.metadata.utils.build_metadata
```

Variables and relevant paths are set on the top of the build_metadata script. 

## Pose extraction 
To perform pose extraction we use the PoseFormat library 

## Pose convertion 

## Segmentation
TBI.

## Handedness 
### Evaluate number of signing hands
To evaluate wheter one or two hands are signing 

``` Evaluate number of signing hands 
python -m PoseTools.handedness.evaluate_hand_number
```

Module to evaluate wheter the right, left or both hands are signing. To run module:

``` Evaluate Handedness
python -m PoseTools.handedness.evaluate_handedness
```

The estimation is based on differences in the integrated motion of the Center-of-Mass of each individual hand. 
