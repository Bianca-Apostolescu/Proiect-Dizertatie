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
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from torcheval.metrics import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from torchmetrics.classification import Dice, BinaryJaccardIndex
# from torchmetrics.detection import IntersectionOverUnion
# from torchmetrics import JaccardIndex


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# %matplotlib inline


def test_model(model, dataloader, loss_function, device, channels, dataset_type):
    print("Testing...")


    conf_matrix = 0
    totalTestLoss = 0
    accuracy, recall, precision, f1_score, dice_score, iou = 0, 0, 0, 0, 0, 0 


    model.eval()
    with torch.no_grad():
        
        if dataset_type == 'cocoms':

          for orig_images, binary_masks, masks in dataloader:
              orig_images, binary_masks, masks = orig_images.to(device), binary_masks.to(device), masks.to(device)
              
              if channels == 3:
                pred_masks = model(orig_images)
              
              # input_tensor = torch.cat([orig_images, altered_images], dim=1) # channel
              # pred_masks = model(input_tensor)
              # pred_masks = model(altered_images) # Testing for altered images 
              
              # tTransform both masks into binary - just to be sure 
              masks = (masks > 0.5).float()
              pred_masks = (pred_masks > 0.5).float()

              # Check if they are binary
              # print(f"binary masks = {((masks == 0) | (masks == 1)).all()}")
              # print(f"binary pred masks = {((pred_masks == 0) | (pred_masks == 1)).all()}")
              
              test_loss = loss_function(pred_masks, masks)
              totalTestLoss += test_loss.item()

              # Check if tensors
              # print("masks is a PyTorch tensor." if torch.is_tensor(masks) else "masks is not a PyTorch tensor.")
              # print("pred_masks is a PyTorch tensor." if torch.is_tensor(pred_masks) else "pred_masks is not a PyTorch tensor.")


              # Apply Green Screen Effects to Predicted Masks
              ## Select a random background from the background folder 
              bg_path = '/content/backgrounds'  # Replace with your actual path

              # List all files in the background_path directory
              background_images = [os.path.join(bg_path, file) for file in os.listdir(bg_path)
                                  if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))]

              # Check if there are any images in the folder
              if not background_images:
                  raise FileNotFoundError("No background images found in the specified folder.")

              # Select a random background image from the list
              random_background_path = random.choice(background_images)

              # Load the random background image using OpenCV
              background_img = cv2.imread(random_background_path)

              # Check if the image was loaded correctly
              if background_img is None:
                  raise ValueError(f"Failed to load the image from {random_background_path}")

              print(f"Random background image selected: {random_background_path}")


              ## Actually apply the background image to the predicted masks
              gs_image = gs.apply_gs(orig_images, pred_masks, background_img)

              # Plot results - images 
              print('\n')
              lent = orig_images.cpu().numpy().shape[0]
              pr.plot_results(lent, orig_images, binary_masks, masks, pred_masks, gs_image)

              # Flatten the masks tensors
              masks = masks.view(-1)
              pred_masks = pred_masks.view(-1)

              # Torch Metrics
              metric = BinaryAccuracy()
              metric.update(pred_masks, masks)
              accuracy += metric.compute()

              metric = BinaryPrecision()
              metric.update(pred_masks, masks)
              precision += metric.compute()

              metric = BinaryRecall()
              metric.update(pred_masks.to(torch.uint8), masks.to(torch.uint8))
              recall += metric.compute()

              metric = BinaryF1Score()
              metric.update(pred_masks, masks)
              f1_score += metric.compute()

              metric = BinaryJaccardIndex().to(device)
              metric.update(pred_masks, masks)
              iou += metric.compute()

              metric = Dice().to(device)
              metric.update(pred_masks.to(device), masks.long().to(device))
              dice_score += metric.compute()

            
            
    avg_test_loss   = totalTestLoss / len(dataloader)
    avg_accuracy    = accuracy / len(dataloader)
    avg_precision   = precision / len(dataloader)
    avg_recall      = recall / len(dataloader)
    avg_f1_score    = f1_score / len(dataloader)
    avg_dice_score  = dice_score / len(dataloader)
    avg_iou         = iou / len(dataloader)

    return avg_test_loss, avg_accuracy, avg_precision, avg_recall, avg_f1_score, avg_dice_score, avg_iou

