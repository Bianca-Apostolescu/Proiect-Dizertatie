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

# Performance Metrics
from sklearn.metrics import multilabel_confusion_matrix


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# %matplotlib inline


# In[ ]:





# In[ ]:


def main_loop(loaded_df, test_paths, transforms_train, transforms_test, wb_name, lr, batch_size, epochs, test_split, valid_split, pin_memory):
    
    wandb.login()
    
    imagePaths = list(loaded_df["Image_Paths"])
    binaryMaskPaths = list(loaded_df["Binary_Paths"])

    for tts in test_split:
        print("[INFO] TEST_SPLIT = {} ...".format(tts))

        print("Splits, Datasets, and Dataloaders")
        startTime = time.time()

        split = train_test_split(imagePaths, binaryMaskPaths, test_size = tts, random_state = 19)
        (trainImages, testImages) = split[:2]
        (trainMasks, testMasks) = split[2:]

        split = train_test_split(trainImages, trainMasks, test_size = valid_split, random_state = 19)
        (trainImages, valImages) = split[:2]
        (trainMasks, valMasks) = split[2:]

        print("[INFO] saving testing image paths...")
        f = open(test_paths, "w")
        f.write("\n".join(testImages))
        f.close()

       
        # Create datasets and data loaders for training, validation, and testing sets
        train_dataset = crd.SegmentationDataset(trainImages, trainMasks, transforms = transforms_train)
        val_dataset   = crd.SegmentationDataset(valImages,   valMasks,   transforms = transforms_test)
        test_dataset  = crd.SegmentationDataset(testImages,  testMasks,  transforms = transforms_test)

        train_loader = DataLoader(train_dataset, shuffle = True,  batch_size = batch_size, pin_memory = pin_memory)
        val_loader   = DataLoader(val_dataset,   shuffle = False, batch_size = batch_size, pin_memory = pin_memory)
        test_loader  = DataLoader(test_dataset,  shuffle = False, batch_size = batch_size, pin_memory = pin_memory)
        


        endTime = time.time()
        print("[INFO] Total time taken to create the dataset and dataloader: {:.2f}s".format(endTime - startTime))

        # calculate steps per epoch for training set
        trainSteps = len(train_dataset) // batch_size
        testSteps  = len(test_dataset) // batch_size
        valSteps   = len(val_dataset) // batch_size

        print(f"trainSteps = {trainSteps}, testSteps = {testSteps}, valSteps = {valSteps}")

        for epoch in epochs:
            
            unet = smp.Unet(
                encoder_name = "resnet101",
                encoder_weights = "imagenet",
                in_channels = 3,  # 3 channels for the image
                classes = 1,  # 1 class => binary mask
                activation = 'sigmoid'
               ).to(device)
            
            model = unet
            
            # Initialize loss function and optimizer
            # lossFunc = nn.BCEWithLogitsLoss()
            lossFunc = dcloss.DiceBCELoss()
            opt = torch.optim.Adam(model.parameters(), lr = lr)

            wandb.init(
              project = wb_name,
              name = "init_metrics_run_" + "tts" + str(tts) + "_ep" + str(epoch), 
              # Track hyperparameters and run metadata
              config = {
                      "learning_rate": lr,
                      "epochs": epochs,
                      "batch": batch_size
                      },
              )
            
            
            print("[INFO] Training the network for {} epochs...".format(epoch))
            
            
            startTime = time.time()

            for e in tqdm(range(epoch)):
                
                #### TRAINING LOOP ####
                avg_train_loss = trModel.train_model(model, train_loader, lossFunc, opt, device)


                #### VALIDATION LOOP ####
                avg_val_loss = valModel.validate_model(model, val_loader, lossFunc, device)

                early_stopping = stopping.EarlyStopping(patience = 5, verbose = True)

                # Check if validation loss has improved
                early_stopping(avg_val_loss)

                # If validation loss hasn't improved, break the loop
                if early_stopping.early_stop:
                    print("Early stopping")

                # Log the losses to WandB
                wandb.log(
                        {
                        "Epoch": e,
                        "Train Loss": avg_train_loss,
                        "Valid Loss": avg_val_loss,
                        }
                        )


            # Display total time taken to perform the training
            endTime = time.time()
            print("[INFO] Total time taken to train and validate the model: {:.2f}s".format(endTime - startTime))


            #### TESTING LOOP ####
            avg_test_loss, avg_accuracy, avg_precision, avg_recall, avg_f1_score, avg_dice_score, avg_iou = testModel.test_model(model, test_loader, lossFunc, device)

            print(f"avg_accuracy = {avg_accuracy}, avg_precision = {avg_precision}, avg_recall = {avg_recall}, avg_f1_score = {avg_f1_score}, avg_dice_score = {avg_dice_score}, avg_iou = {avg_iou}")

            wandb.log(
                    {
                    "Accuracy": avg_accuracy,
                    "Precision": avg_precision,
                    "Recall": avg_recall,
                    "F1-Score": avg_f1_score,
                    "DICE": avg_dice_score,
                    "IOU": avg_iou,
                    }
                    )

