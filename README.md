# A Deep Learning Method to Detect Dogs' Heart Disease from Simple X-Rays: Applying an improved attention U-Net for size-unbalanced regions of interest

## Create time : 2022. 01. 09

## Introduction
This repository contains the source code for developing a deep learning-based semantic segmentation model with canine chest x-ray. We used [improved attention u-net](https://github.com/nabsabraham/focal-tversky-unet) Model was implemented with Keras API.

## Installation 
- pip install -r requirements.txt 

## Dataset Preparation
- In this study, we genersted ground truth mask for heart and t4 with [labelme](https://github.com/wkentaro/labelme) tool. This program will store `label.png` as the result of annotation. You can convert this label file to binary ground truth mask with `image_module.py` from utils repository.
- example:
```
from utils.image_module import *

# Make HEART only binary mask from label.png (generated from label.png)
label_to_heart_mask(/data, /data/mask_heart, input_size) 

# Make T4 only binary mask from label.png (generated from label.png)
label_to_t4_mask(/data, /data/mask_t4, input_size) 
```

## Dataset Form
```
|-data
    |--input
        |--1.jpg
        |--2.jpg
        ...
    |--mask # generated from labelme tool 
        |--1.png
        |--2.png
        ...
    |--mask_heart # converted by label_to_heart_mask
        |--1.jpg
        |--2.jpg
        ...
    |--mask_t4 # converted by label_to_t4_mask
        |--1.jpg
        |--2.jpg
        ...
```

# Training 
- Training is executed as follows:
```
python3 train.py --data DATA_PATH --weight-path WEIGHT_PATH --plot-path PLOT_PATH --test-plot-path TEST_PLOT_PATH
```
- If you want to infer test results, you can use 2 options
  - set `--test True` : It will show precision-recall curve, dice, precision and recall score for test set.
  - set `--test-plot True` : It will save test result plots (original image / ground truth / predicted mask) 
