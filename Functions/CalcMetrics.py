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


# In[ ]:





# In[ ]:


def calculate_metrics(gt_masks, pred_masks):
    tp = 0  # true positive
    tn = 0  # true negative
    fp = 0  # false positive
    fn = 0  # false negative

    for gt_mask, pred_mask in zip(gt_masks, pred_masks):
        gt_mask = gt_mask.flatten()
        pred_mask = pred_mask.flatten()

        tp += np.sum(np.logical_and(pred_mask == 1, gt_mask == 1))
        tn += np.sum(np.logical_and(pred_mask == 0, gt_mask == 0))
        fp += np.sum(np.logical_and(pred_mask == 1, gt_mask == 0))
        fn += np.sum(np.logical_and(pred_mask == 0, gt_mask == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_score

def calculate_dice_coefficient(gt_masks, pred_masks):
    intersection = np.logical_and(gt_masks, pred_masks)
    dice_coefficient = (2.0 * intersection.sum()) / (gt_masks.sum() + pred_masks.sum())
    return dice_coefficient

def calculate_iou(gt_masks, pred_masks):
    intersection = np.logical_and(gt_masks, pred_masks)
    union = np.logical_or(gt_masks, pred_masks)
    iou = intersection.sum() / union.sum()
    return iou

