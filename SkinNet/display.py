#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 10:05:55 2023

Display image 
@author: julien
"""

from matplotlib import pyplot as plt
import os
import random
from tensorflow.keras.utils import load_img
import numpy as np
from PIL import Image
import cv2


def print_random_figures_from_folder(img_folder,mask_folder):

    plt.figure(figsize=(20,20))
    for i in range(5):
        image_file = random.choice(os.listdir(img_folder))
        image_path= os.path.join(img_folder, image_file)
        img=load_img(image_path)
        
        ax=plt.subplot(2,5,i+1)
        ax.title.set_text(image_file)
        plt.imshow(img)
        
        filename = image_file.split('.')[0] + '.pbm'
        mask_path= os.path.join(mask_folder,filename)
        img=load_img(mask_path)
        ax=plt.subplot(2,5,i+6)
        ax.title.set_text(image_file)
        plt.imshow(img)
        
    image= np.array(load_img(image_path))
    return image
        
        
def create_dataset_PIL_with_classes(img_folder,IMG_HEIGHT,IMG_WIDTH):
    
    img_data_array=[]
    class_name=[]
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
       
            image_path= os.path.join(img_folder, dir1,  file)
            image= Image.open(image_path)
            image= image.resize((IMG_HEIGHT,IMG_WIDTH,3))
            image= np.array(image)
            image = image.astype('float32')
            image /= 255  
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array , class_name

def create_narray(img_folder,mask_folder,im_size = 128):
    
    img_data_array=[]
    mask_data_array = []
    for file in os.listdir(img_folder):
        if not file.startswith('.'):
            #get image
            image_path= os.path.join(img_folder,file)
            image= Image.open(image_path)
            image= image.resize((im_size,im_size))
            image= np.array(image)
            image = image.astype('float32')
            image /= 255  
            img_data_array.append(image)
            
            #get mask
            filename = file.split('.')[0] + '.pbm'
            mask_path= os.path.join(mask_folder,filename)
            image= Image.open(mask_path)
            image= image.resize((im_size,im_size))
            image= np.array(Image.open(mask_path))
            image= np.resize(image,(128,128,1))
            mask_data_array.append(image)
            
    img_data_array = np.array(img_data_array, np.float32)
    mask_data_array = np.array(mask_data_array, np.int0)
    
    return (img_data_array, mask_data_array)

def create_narray_resize(img_folder,mask_folder,im_size):
    
    img_data_array=[]
    mask_data_array = []
    for file in os.listdir(img_folder):
        if not file.startswith('.'):
            #get image
            image_path= os.path.join(img_folder,file)
            img=load_img(image_path)
            img= np.array(img)
            image= cv2.resize(img,(im_size,im_size))
            image = image.astype('float32')
            image /= 255  
            img_data_array.append(image)
            
            #get mask
            filename = file.split('.')[0] + '.pbm'
            mask_path= os.path.join(mask_folder,filename)
            img=load_img(mask_path,color_mode="grayscale")        
            img= np.array(img)            
            image= cv2.resize(img,(128,128))
            image = image.astype('float32')
            image /= 255.0
            mask_data_array.append(image)
            
    img_data_array = np.array(img_data_array, np.float32)
    mask_data_array = np.array(mask_data_array, np.int0)
    mask_data_array = np.expand_dims(mask_data_array, 3)
    
    return (img_data_array, mask_data_array)