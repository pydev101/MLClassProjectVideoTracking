# Crowd Density Estimation

## Overview

This repo contains a Python implementation of **CSRNet** compatible with checkpoints from [CSRNet-pytorch](https://github.com/leeyeehoo/CSRNet-pytorch/tree/master)

Output types should be trivial to determine since I added typing

**Current features**
- Load and preprocess RGB images for the network (ImageNet mean/std, batched tensors).
- Load pretrained CSRNet weights and run inference to get a density map and an estimated count (sum of the map).
- Load ShanghaiTech-style ground truth from `.mat` files (head locations) and build a smoothed density map for comparison or visualization.
- Optionally blur density maps
- Plot predicted density, ground-truth density, or a generic overlay on the original image (`plotting_tools.py`);


## Installation
```pip install -r requirements.txt```

The models should be downloaded from the git repo and placed in the models/csr_net_base folder. Delete the .tar extension. it's not actually a tarball. Idk why it's labeled like that.

Note: This is tested and working on Python 3.11.13 

## Example
```python
from pathlib import Path

from csrnet import load_and_process_image_with_csrnet, load_csrnet_model, load_ground_truth
from plotting_tools import plot_csrnet_density_map, plot_ground_truth_density


def example():
    weights_a = Path("./models/csr_net_base/PartBmodel_best.pth")
    img_path = Path("./ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data/images/IMG_2.jpg")

    if not img_path.exists():
        print(f"Warning: Could not find {img_path}. Make sure the image exists to run the example.")
        return

    model = load_csrnet_model(weights_a, map_location="cpu")
    img, density_map, estimated_crowd_count = load_and_process_image_with_csrnet(model, img_path)

    gt_path = img_path.parent.parent / "ground_truth" / f"GT_{img_path.stem}.mat"
    if gt_path.exists():
        gt_density = load_ground_truth(gt_path, image_path=img_path)
        plot_ground_truth_density(img, gt_density)

    plot_csrnet_density_map(img, density_map, estimated_crowd_count)
```

## Shanghai Tech Density Dataset
A dataset of labeled single crowd images split into two folders containing
- images: the jpg image file
- ground-truth: matlab file containing head annotations (coordinate x, y)
- ground-truth-h5: people density map 
- part_A contains dense crowd images
- part_B contains sparse crowd images

## CRSNet
congested scene recognition (CSRNET): a CNN that accepts images of arbitrary size input and outputs a 2D crowd density estimation map

Roughly, it hsa the following architecture: It is basically VGG-16 without the fully connected classifier, plus a dilated-convolution back end. The front end uses the first 10 convolution layers of VGG-16 with 3 max-pooling layers, so its feature map is 1/8 of the input resolution. The back end processes that output with six dilated 3x3 convolutions. The final result is an output with the same size as the input.

VGG: An image focused CNN architecture that uses lots of small 3x3 convolutions with occasional max pooling layers
Max Pooling Layers: Reduce dimensionality by selecting the largest number in a grid
