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
import GCANet as gca

# Performance Metrics
from sklearn.metrics import multilabel_confusion_matrix


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# %matplotlib inline




def main_loop(loaded_df, test_paths, transforms_train, transforms_test, wb_name, lr, batch_size, epochs, test_split, valid_split, pin_memory, model_type, channels, dataset_type, saved_model_path = None):
    
    wandb.login()
    
    imagePaths = list(loaded_df["Image_Paths"])
    binaryMaskPaths = list(loaded_df["Binary_Paths"])
    # maskPaths = list(loaded_df["Mask_Paths"])

    print(f"images = {len(imagePaths)}")
    print(f"binaryMaskPaths = {len(binaryMaskPaths)}")
    # print(f"maskPaths = {len(maskPaths)}")

    for tts in test_split:
        print("[INFO] TEST_SPLIT = {} ...".format(tts))

        print("Splits, Datasets, and Dataloaders")
        startTime = time.time()


        if dataset_type == 'cocoms':
          split = train_test_split(imagePaths, binaryMaskPaths, test_size = tts, random_state = 19)
          (trainImages, testImages) = split[:2]             # First two elements are the train and test splits of imagePaths
          # (trainBinaryMasks, testBinaryMasks) = split[2:4]  # Next two elements are the train and test splits of binaryMaskPaths
          (trainMasks, testMasks) = split[2:]               # Last two elements are the train and test splits of maskPaths

          split = train_test_split(trainImages, trainMasks, test_size = valid_split, random_state = 19)
          (trainImages, valImages) = split[:2]             # First two elements are the train and test splits of imagePaths
          # (trainBinaryMasks, valBinaryMasks) = split[2:4]  # Next two elements are the train and test splits of binaryMaskPaths
          (trainMasks, valMasks) = split[2:]               # Last two elements are the train and test splits of maskPaths

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
            
            if model_type == 'GCA':
              gcanet = gca.GCANet(in_c = channels, out_c = 1, only_residual = True).to(device)
              model = gcanet

            elif model_type == 'resnet':

              model = smp.Unet(
                  encoder_name = "resnet101",
                  encoder_weights = "imagenet",
                  in_channels = 3,  # 3 channels for the image
                  classes = 1,  # 1 class => binary mask
                  activation = 'sigmoid'
                ).to(device)
              
              # model = resnet

            elif model_type == 'mobilenetv2':
    
              model = smp.Unet(
                  encoder_name = 'mobilenet_v2',  # Use MobileNetV2 as encoder
                  encoder_weights = 'imagenet',  # Use pre-trained weights
                  in_channels = 3,  # 3 channels for the image
                  classes = 1,  # 1 class => binary mask
                  activation = 'sigmoid'
              ).to(device)

              # model = mobilenet

            elif model_type == 'efficientnet':
    
              model = smp.Unet(
                  encoder_name = 'efficientnet-b3',  # Use efficientnet-b3 as encoder
                  encoder_weights = 'imagenet',  # Use pre-trained weights
                  in_channels = 3,  # 3 channels for the image
                  classes = 1,  # 1 class => binary mask
                  activation = 'sigmoid'
              ).to(device)

              # model = efficientnet

            elif model_type == 'deeplab':

              model = smp.DeepLabV3Plus(
                  encoder_name = 'mobilenet_v2',  # Use efficientnet-b3 as encoder
                  encoder_weights = 'imagenet',  # Use pre-trained weights
                  in_channels = 3,  # 3 channels for the image
                  classes = 1,  # 1 class => binary mask
                  activation = 'sigmoid'
              ).to(device)

              # model = deeplab
            
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

            
            if saved_model_path:
              print(f"[INFO] Loading model from {saved_model_path}")
              checkpoint = torch.load(saved_model_path, map_location=device)
              model.load_state_dict(checkpoint['model_state_dict'])

              print("[INFO] Testing the loaded model...")
              avg_test_loss, avg_accuracy, avg_precision, avg_recall, avg_f1_score, avg_dice_score, avg_iou = testModel.test_model(model, test_loader, lossFunc, device, channels, dataset_type)

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

              return
            
            
            print("[INFO] Training the network for {} epochs...".format(epoch))
            
            
            startTime = time.time()

            for e in tqdm(range(epoch)):
                
                #### TRAINING LOOP ####
                avg_train_loss = trModel.train_model(model, train_loader, lossFunc, opt, device, channels, dataset_type)


                #### VALIDATION LOOP ####
                avg_val_loss = valModel.validate_model(model, val_loader, lossFunc, device, channels, dataset_type)

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


            # Save models 
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, f'epoch_{epoch}_model.pth'))
            # model.save(os.path.join(wandb.run.dir, f'epoch_{epoch}_model.pth'))
            wandb.save(f'epoch_{epoch}_model.pth')

            #### TESTING LOOP ####
            avg_test_loss, avg_accuracy, avg_precision, avg_recall, avg_f1_score, avg_dice_score, avg_iou = testModel.test_model(model, test_loader, lossFunc, device, channels, dataset_type)

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

