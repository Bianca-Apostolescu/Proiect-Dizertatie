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


def apply_gs(orig_image, pred_masks, background):
    
    # Resize the new background to match the original image size
    new_background = cv2.resize(new_background, (orig_image.shape[1], orig_image.shape[0]))

    # Ensure the mask is binary (0 and 255 values)
    _, binary_mask = cv2.threshold(pred_masks, 128, 255, cv2.THRESH_BINARY)

    # Invert the mask to get the background mask
    inverse_mask = cv2.bitwise_not(binary_mask)

    # Convert masks to 3 channels to match the image dimensions
    binary_mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    inverse_mask_3ch = cv2.cvtColor(inverse_mask, cv2.COLOR_GRAY2BGR)

    # Extract the foreground from the original image
    foreground = cv2.bitwise_and(orig_image, binary_mask_3ch)

    # Extract the corresponding region from the new background
    background = cv2.bitwise_and(new_background, inverse_mask_3ch)

    # Combine the foreground with the new background
    result_image = cv2.add(foreground, background)

    # Convert images from BGR to RGB for correct color display with matplotlib
    original_image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    return result_image_rgb


