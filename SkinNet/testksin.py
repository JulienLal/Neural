#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 12:03:15 2023

@author: julien
"""
from display import print_random_figures_from_folder,create_narray
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import PIL
import cv2
test_image_dir = "../../Datasets/SkinDataset/Test/skin-images/"
test_mask_dir = "../../Datasets/SkinDataset/Test/masks/"

Image = print_random_figures_from_folder(test_image_dir,test_mask_dir)
Image_size = Image.shape


Image_resized = cv2.resize(Image, (257,257))




                     

Image_resized = Image_resized[np.newaxis,:]

# Restore the model.

model = tf.keras.models.load_model('best_model.hdf5')


prediction = model.predict(Image_resized)


prediction = prediction.squeeze()
prediction_resized = cv2.resize(prediction,(Image_size[1],Image_size[0]))
threshold = 0.95
prediction_thres = prediction_resized>threshold
prediction_thres = prediction_thres.astype(int)

ax=plt.subplot(1,3,1)
ax.title.set_text("Image")
plt.imshow(Image)
ax=plt.subplot(1,3,2)
ax.title.set_text("prediction_resized")
plt.imshow(prediction_resized)
ax=plt.subplot(1,3,3)
ax.title.set_text("prediction_thres: {}".format(threshold))
plt.imshow(prediction_thres)

"""
plt.figure(figsize=(20,20))
plt.imshow(prediction.squeeze())
prediction_resized = prediction
prediction_resized.resize((1,Image_size[0],Image_size[1],1),refcheck=False)

plt.figure(figsize=(20,20))
plt.imshow(prediction_resized.squeeze()*255.0)
"""