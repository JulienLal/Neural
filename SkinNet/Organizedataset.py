#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:25:55 2023

@author: julien
"""

import os
import random 
import shutil
import pathlib

# Extracting data
data_dir = pathlib.Path("../../Datasets/SkinDataset")
masks_dir = pathlib.Path("../../Datasets/SkinDataset/masks")
images_dir = pathlib.Path("../../Datasets/SkinDataset/skin-images")


tot_images = len(os.listdir('../../Datasets/SkinDataset/masks'))
print("Dataset is {} images".format(tot_images))


#Get list of masks
mask_count_pmb = len(list(data_dir.glob('*/*.pbm')))
files_list = list(data_dir.glob('*/*.pbm'))

#Pick up mask in list define 10% ratio for validation and testing
nb_validation_set = int(tot_images*10/100)
print("Number of images taken as validation and test: {}".format(nb_validation_set*2))

list_random = random.sample(range(0,tot_images),nb_validation_set*2)

#Validation dataset
for i in list_random[:nb_validation_set]:
    file_name = files_list[i].stem
    for fname in os.listdir('../../Datasets/SkinDataset/skin-images'):    # change directory as needed
        image_name = fname.split(".")[0]
        if file_name == image_name:
            shutil.move(str(files_list[i]), "../../Datasets/SkinDataset/Validation/masks")
            shutil.move(str('../../Datasets/SkinDataset/skin-images/'+fname), "../../Datasets/SkinDataset/Validation/skin-images")

#Test dataset
for i in list_random[nb_validation_set:]:
    file_name = files_list[i].stem
    for fname in os.listdir('../../Datasets/SkinDataset/skin-images'):    # change directory as needed
        image_name = fname.split(".")[0]
        if file_name == image_name:
            shutil.move(str(files_list[i]), "../../Datasets/SkinDataset/Test/masks")
            shutil.move(str('../../Datasets/SkinDataset/skin-images/'+fname), "../../Datasets/SkinDataset/Test/skin-images")

