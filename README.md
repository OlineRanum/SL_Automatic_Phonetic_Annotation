# PoseTools

This repository contains code to process poses, with functionalities for automatically annotating phonetic attributes of sign languages


### Handedness 
Module to evaluate wheter the right, left or both hands are signing. To run module:

``` Evaluate Handedness
python -m PoseTools.handedness.evaluate_handedness
```

The estimation is based on differences in the integrated motion of the Center-of-Mass of each individual hand. 
