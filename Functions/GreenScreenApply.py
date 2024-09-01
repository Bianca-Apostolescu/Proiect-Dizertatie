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
    Apply the green screen effect to a single image.

    Args:
        orig_image (numpy.ndarray): Original image of shape (H, W, C) in BGR format
        pred_mask (numpy.ndarray): Binary mask of shape (H, W)
        background (numpy.ndarray): Background image of shape (H, W, 3) in BGR format

    Returns:
        numpy.ndarray: Image with the green screen effect applied.
    """
    # Resize the background to match the original image size
    resized_background = cv2.resize(background, (orig_image.shape[1], orig_image.shape[0]))

    # Ensure the mask is binary (0 and 255 values)
    _, binary_mask = cv2.threshold(pred_mask, 128, 255, cv2.THRESH_BINARY)

    # Invert the mask to get the background mask
    inverse_mask = cv2.bitwise_not(binary_mask)

    # Convert masks to 3 channels to match the image dimensions
    binary_mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    inverse_mask_3ch = cv2.cvtColor(inverse_mask, cv2.COLOR_GRAY2BGR)

    # Convert all images and masks to uint8
    orig_image = orig_image.astype(np.uint8)
    binary_mask_3ch = binary_mask_3ch.astype(np.uint8)
    resized_background = resized_background.astype(np.uint8)
    inverse_mask_3ch = inverse_mask_3ch.astype(np.uint8)

    # Debug print to check sizes and data types
    # print(f"Original image dtype: {orig_image.dtype}")
    # print(f"Binary mask dtype: {binary_mask_3ch.dtype}")
    # print(f"Resized background dtype: {resized_background.dtype}")
    # print(f"Inverse mask dtype: {inverse_mask_3ch.dtype}")

    # Ensure the foreground and background images have the same size
    if orig_image.shape != binary_mask_3ch.shape:
        raise ValueError(f"Size mismatch: orig_image {orig_image.shape}, binary_mask_3ch {binary_mask_3ch.shape}")
    if resized_background.shape != inverse_mask_3ch.shape:
        raise ValueError(f"Size mismatch: resized_background {resized_background.shape}, inverse_mask_3ch {inverse_mask_3ch.shape}")

    # Extract the foreground from the original image
    foreground = cv2.bitwise_and(orig_image, binary_mask_3ch)

    # Extract the corresponding region from the new background
    background_region = cv2.bitwise_and(resized_background, inverse_mask_3ch)

    # Combine the foreground with the new background
    result_image = cv2.add(foreground, background_region)

    return result_image


