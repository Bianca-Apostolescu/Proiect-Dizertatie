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

# Performance Metrics
from sklearn.metrics import multilabel_confusion_matrix


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# %matplotlib inline



def filter_data(ann_path, filterCls, bm_path, IMAGE_PATH, MASK_PATH):
    
    # Lists of Person Images and Masks
    imagePaths = []
    maskPaths = []
    binaryMaskPaths = []
    
    # Initialize the COCO api for instance annotations
    coco = COCO(ann_path)

    # Fetch class IDs only corresponding to the filterClasses
    catIds = coco.getCatIds(catNms = filterCls) 

    # Get all images containing the above Category IDs
    imgIds = coco.getImgIds(catIds = catIds)
    print("Number of images containing all the classes:", len(imgIds))
    

    for i in range (0, len(imgIds)):

        if i % 1000 ==0:
            print(i)

        img = coco.loadImgs(imgIds[i])[0]


        annIds = coco.getAnnIds(imgIds = img['id'], catIds = catIds, iscrowd = None)
        anns = coco.loadAnns(annIds)

        bin_mask = bm.gen_binary_mask(img, coco, anns)
        bin_mask = (bin_mask * 255).astype(np.uint8)
        bin_mask = Image.fromarray(bin_mask) # Convert the array to a PIL Image

        if bin_mask.mode != 'L':
            bin_mask = bin_mask.convert('L')

        bin_mask.save(bm_path + img['file_name'][:12] + '.png') # Save the mask image to disk


        imagePaths.append((IMAGE_PATH + '/' + img['file_name']))
        maskPaths.append((MASK_PATH + '/' + img['file_name'][:12] + '.png'))
        binaryMaskPaths.append(bm_path + img['file_name'][:12] + '.png')
        
    return imagePaths, maskPaths, binaryMaskPaths

