#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# For managing COCO dataset
from pycocotools.coco import COCO

# For creating and managing folder/ files
import glob
import os
import shutil

# For managing images
from PIL import Image
import skimage.io as io

# Basic libraries
import numpy as np
import pandas as pd
import random
import cv2

# For plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# For importing models and working with them
## Torch
import torch
import torch.utils.data # for Dataset
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

## Torchvision
import torchvision
from torchvision.transforms import transforms

# For creating train - test splits
from sklearn.model_selection import train_test_split

import pathlib
import pylab
import requests
from io import BytesIO
from pprint import pprint
from tqdm import tqdm
import time
from imutils import paths

# Performance Metrics
from sklearn.metrics import multilabel_confusion_matrix

# Import Files
import BinaryMasks as bm
import CalcMetrics as cm
import CreateDataset as crd
import FilterImgs as flt
import TrainModel as trModel
import ValidateModel as valModel
import TestModel as testModel
import SaveCSV
import DiceLoss as dcloss
import EarlyStopping as stopping
import PlotResults as pr
import MainLoop as main
import DisplayMetrics as dm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# %matplotlib inline

def apply_gs(orig_image, pred_mask, background):
    """
    Apply a green screen effect using the predicted mask to overlay a new background onto the original image.

    Args:
        orig_image (numpy array or torch tensor): The original image, either as a numpy array or PyTorch tensor.
        pred_mask (numpy array or torch tensor): The predicted mask, either as a numpy array or PyTorch tensor.
        background (numpy array): The new background image.

    Returns:
        numpy array: A single image with the green screen effect applied.
    """
    # Convert inputs to numpy arrays if they are PyTorch tensors
    if isinstance(orig_image, torch.Tensor):
        orig_image = orig_image.cpu().numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()

    # Select the first image and mask from the batch
    orig_image = orig_image[0]
    pred_mask = pred_mask[0]

    # fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    # axes[0].imshow(orig_image)
    # axes[0].set_title("Original Image")
    # axes[0].axis('off')

    # axes[1].imshow(pred_mask, cmap='gray')
    # axes[1].set_title("Segmentation Mask")
    # axes[1].axis('off')

    # plt.tight_layout()
    # plt.show()

    # Resize the background to match the original image size
    background_resized = cv2.resize(background, (orig_image.shape[2], orig_image.shape[1]))

    # Ensure the mask is binary (0 and 255 values)
    pred_mask = (pred_mask * 255).astype(np.uint8)  # Convert to uint8 for OpenCV functions
    if len(pred_mask.shape) == 3 and pred_mask.shape[0] == 1:
        pred_mask = pred_mask[0]  # Remove the channel dimension if it's 1

    # Check mask dimensions for single channel
    if len(pred_mask.shape) != 2:
        raise ValueError(f"Expected pred_mask to be a single-channel grayscale image, got shape: {pred_mask.shape}")

    # Invert the mask to get the background mask
    _, binary_mask = cv2.threshold(pred_mask, 128, 255, cv2.THRESH_BINARY)
    inverse_mask = cv2.bitwise_not(binary_mask)

    # Convert masks to 3 channels to match the image dimensions
    binary_mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    inverse_mask_3ch = cv2.cvtColor(inverse_mask, cv2.COLOR_GRAY2BGR)

    # Convert original image to BGR format if needed
    if orig_image.shape[0] == 3:  # RGB format, convert to BGR
        orig_image_bgr = orig_image.transpose(1, 2, 0).astype(np.uint8)
    else:
        orig_image_bgr = cv2.cvtColor(orig_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Ensure that the original image and mask have the same size and type
    if orig_image_bgr.shape != binary_mask_3ch.shape:
        raise ValueError(f"Shape mismatch: orig_image_bgr shape {orig_image_bgr.shape}, binary_mask_3ch shape {binary_mask_3ch.shape}")

    print(f"orig_image_bgr = {orig_image_bgr.shape}")
    print(f"binary_mask_3ch = {binary_mask_3ch.shape}")

    # Extract the foreground from the original image
    foreground = cv2.bitwise_and(orig_image_bgr, binary_mask_3ch)

    # Extract the corresponding region from the new background
    background_region = cv2.bitwise_and(background_resized, inverse_mask_3ch)

    # Combine the foreground with the new background
    result_image = cv2.add(foreground, background_region)

    # Convert the result image from BGR to RGB
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    # Return the processed image
    return result_image_rgb


