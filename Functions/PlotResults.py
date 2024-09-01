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
import wandb

# For importing models and working with them
## Torch
import torch
import torch.utils.data # for Dataset
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp

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
import GreenScreenApply as gs

# Performance Metrics
from sklearn.metrics import multilabel_confusion_matrix


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# %matplotlib inline


def plot_results(lent, orig_images, masks, pred_masks, background_img):
   
    for i in range(lent):  # For each image in the batch
        # Convert tensors to NumPy arrays
        orig_image_np = orig_images.cpu().numpy()[i].transpose(1, 2, 0)  # (C, H, W) to (H, W, C)
        pred_mask_np = pred_masks.cpu().numpy()[i][0]  # Get the single channel mask
        pred_mask_np = (pred_mask_np * 255).astype(np.uint8)  # Convert mask to 0-255 range
        reversed_pred_mask_np = np.logical_not(pred_mask_np).astype(np.uint8) * 255

        # Ensure the background image is resized to match the original image
        background_resized = cv2.resize(background_img, (orig_image_np.shape[1], orig_image_np.shape[0]))

        # Apply the green screen effect
        gs_image_np = gs.apply_gs(orig_image_np, pred_mask_np, background_resized)


        # Log image(s)
        wandb.log(
                {"original_images": [wandb.Image(orig_images.cpu().numpy()[i].transpose(1, 2, 0), caption = "Original_Image")],
                  # "seg_masks": [wandb.Image(binary_masks.cpu().numpy()[i][0], caption = "Seg_Mask")],
                  "gt_masks": [wandb.Image(masks.cpu().numpy()[i][0], caption = "GT_Mask")],
                  "pred_masks": [wandb.Image(pred_masks.cpu().numpy()[i][0], caption = "Pred_Mask")],
                  "gs_images": [wandb.Image(cv2.cvtColor(gs_image_np, cv2.COLOR_BGR2RGB), caption = "GreenScreen_Image")]
                })




          

