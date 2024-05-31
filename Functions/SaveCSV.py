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



def create_csvs(imagePaths, binaryMaskPaths, small, medium, large):
    
    data = {'Image_Paths': imagePaths, 
            'Binary_Paths': binaryMaskPaths,
            # 'Mask_Paths': maskPaths
           }

    df = pd.DataFrame(data = data)
    shuffled_df_new = df.sample(frac = 1, random_state = 49)

    shuffled_df_copy = shuffled_df_new.copy()

    shuffled_df_small = shuffled_df_copy.head(small)
    shuffled_df_medium = shuffled_df_copy.head(medium)
    shuffled_df_large = shuffled_df_copy.head(large)

    shuffled_df_small.to_csv('shuffled_df_small.csv', index=False)
    shuffled_df_medium.to_csv('shuffled_df_medium.csv', index=False)
    shuffled_df_large.to_csv('shuffled_df_large.csv', index=False)
    shuffled_df_copy.to_csv('shuffled_df_all.csv', index=False)

