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

# Correlation and Clustering
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor

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

def resize_image(image_path, width):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (width, width))

    return img

def calculate_image_correlations(image_paths, input_image_width):
   
    num_images = len(image_paths)
    correlations = np.zeros((num_images, num_images))

    # Read and resize all images first
    resized_images = []
    for path in tqdm(image_paths, desc = "Loading and resizing images"):
        resized_images.append(resize_image(path, input_image_width))
    
    # Function to compute the similarity between images
    def compute_similarity(i, j):
        similarity, _ = ssim(resized_images[i], resized_images[j], full=True)
        return similarity

    # Compute correlations using parallel processing
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(num_images):
            for j in range(i + 1, num_images):
                futures.append((i, j, executor.submit(compute_similarity, i, j)))
        
        for i, j, future in futures:
            similarity = future.result()
            correlations[i, j] = similarity
            correlations[j, i] = similarity

    return correlations


def find_correlated_groups(correlations, threshold = 0.95):
    distances = 1 - correlations
    clustering = DBSCAN(eps = 1 - threshold, min_samples = 1, metric = 'precomputed').fit(distances)
    
    return clustering.labels_


def prune_images(image_paths, labels, num_to_remove_fraction = 0.5):
    grouped_images = defaultdict(list)
    
    for img_path, label in zip(image_paths, labels):
        grouped_images[label].append(img_path)
    
    to_remove = set()
    
    for group, images in grouped_images.items():
        num_to_remove = int(len(images) * num_to_remove_fraction)
        to_remove.update(set(images[:num_to_remove]))
    
    return to_remove


def update_dataset(image_folder, binary_folder, third_folder, images_to_remove):
    """
    Remove specified images, their corresponding binary masks, and additional files from a third folder.

    Parameters:
        image_folder (str): Path to the folder containing the original images (.jpg).
        binary_folder (str): Path to the folder containing the binary masks (.png).
        third_folder (str): Path to the third folder containing additional files.
        images_to_remove (set): A set of base names (without extensions) of images to be removed.
    """

    # Removing original images
    for img_name in os.listdir(image_folder):
        if img_name.endswith('.jpg'):
            base_name = os.path.splitext(img_name)[0]  # Extract base name without extension
            if base_name in images_to_remove:
                full_img_path = os.path.join(image_folder, img_name)
                os.remove(full_img_path)
    
    # Removing binary masks
    for mask_name in os.listdir(binary_folder):
        if mask_name.endswith('.png'):
            base_name = os.path.splitext(mask_name)[0]  # Extract base name without extension
            if base_name in images_to_remove:
                full_mask_path = os.path.join(binary_folder, mask_name)
                os.remove(full_mask_path)

    # Removing files from the panoptic folder
    for third_file_name in os.listdir(third_folder):
        base_name, ext = os.path.splitext(third_file_name)  # Extract base name and extension
        if base_name in images_to_remove:
            full_third_file_path = os.path.join(third_folder, third_file_name)
            os.remove(full_third_file_path)

    print(f"Removed {len(images_to_remove)} images, masks, and corresponding files from the third folder.")
