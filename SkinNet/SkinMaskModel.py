#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:31:17 2023

@author: julien
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose


    
    

#Take as input an RGB image and return as output binary image of same size
def encoder(inputs, num_filters):
    
    # 2xConvolution 3x3 followed by Maxppol to reduce syize
    
    x = Conv2D(filters=num_filters,kernel_size = (3,3),padding = 'valid', activation = 'relu') (inputs)
    
    x = Conv2D(filters=num_filters,kernel_size = (3,3),padding = 'valid', activation = 'relu') (x)
    
    x = MaxPool2D(pool_size= (2,2),padding='valid', strides=2 ) (x)
    
    return x
    

def decoder(inputs, merge_feature, num_filters):
    
    # Upsampling     
    x = Conv2DTranspose(filters=num_filters,kernel_size = (2,2), strides = 2) (inputs)
    
    # Copy and crop the skip features 
    # to match the shape of the upsampled input
    skip_features = tf.image.resize(merge_feature,
                                    size = (x.shape[1],
                                            x.shape[2]))
    x = tf.keras.layers.Concatenate()([x, skip_features])
      
    # Two Conv2D
    x = Conv2D(filters=num_filters,kernel_size = (3,3),padding = 'same', activation = 'relu') (x)
    
    x = Conv2D(filters=num_filters,kernel_size = (3,3),padding = 'same', activation = 'relu') (x)
    
        
    return x

def bottleneck(inputs,num_filters):
    
    x = Conv2D(filters=num_filters,kernel_size = 3,padding = 'valid', activation = 'relu') (inputs)

    x = Conv2D(filters=num_filters,kernel_size = 3,padding = 'valid', activation = 'relu') (x)
    
    return x

def unet(input_shape = (257, 257, 3), num_output_classes = 1):
    inputs = tf.keras.layers.Input(input_shape)

    #Contracting oath
    c1 = encoder(inputs,64)
    c2 = encoder(c1,128)
    c3 = encoder(c2,256)
    c4 = encoder(c3,512)

    #Bottleneck
    b = bottleneck(c4, 1024)(c4)

    
    #Expensive path
    e1 = decoder(b, c4, 512)
    e2 = decoder(e1, c3, 216)
    e3 = decoder(e2, c2, 128)
    e4 = decoder(e3, c2, 64)
    
    
    
    
    
    return model

def Unet_model(input_shape = (257, 257, 3)):
    
    inputs = tf.keras.layers.Input(input_shape)
    
    # First encoder
    x = Conv2D(filters=64,kernel_size = (3,3),padding = 'valid', activation = 'relu') (inputs)
    e1 = Conv2D(filters=64,kernel_size = (3,3),padding = 'valid', activation = 'relu') (x)
    ouput_e1 = MaxPool2D(pool_size= (2,2),padding='valid', strides=2 ) (e1)
    
    # second encoder
    x = Conv2D(filters=128,kernel_size = (3,3),padding = 'valid', activation = 'relu') (ouput_e1)
    e2 = Conv2D(filters=128,kernel_size = (3,3),padding = 'valid', activation = 'relu') (x)
    ouput_e2 = MaxPool2D(pool_size= (2,2),padding='valid', strides=2 ) (e2)
    
    # third encoder
    x = Conv2D(filters=256,kernel_size = (3,3),padding = 'valid', activation = 'relu') (ouput_e2)
    e3 = Conv2D(filters=256,kernel_size = (3,3),padding = 'valid', activation = 'relu') (x)
    ouput_e3 = MaxPool2D(pool_size= (2,2),padding='valid', strides=2 ) (e3)
     
    # fourth encoder
    x = Conv2D(filters=512,kernel_size = (3,3),padding = 'valid', activation = 'relu') (ouput_e3)
    e4 = Conv2D(filters=512,kernel_size = (3,3),padding = 'valid', activation = 'relu') (x)
    ouput_e4 = MaxPool2D(pool_size= (2,2),padding='valid', strides=2 ) (e4)
   
    # bottleneck
    x = Conv2D(filters=1024,kernel_size = 3,padding = 'valid', activation = 'relu') (ouput_e4)
    x = Conv2D(filters=1024,kernel_size = 3,padding = 'valid', activation = 'relu') (x)
   
    # fist decoder
    x = Conv2DTranspose(filters=512,kernel_size = (2,2), strides = 2) (x)   
    skip_features = tf.image.resize(e4, size = (x.shape[1], x.shape[2]))
    x = tf.keras.layers.Concatenate()([x, skip_features])
    x = Conv2D(filters=512,kernel_size = (3,3),padding = 'same', activation = 'relu') (x)
    x = Conv2D(filters=512,kernel_size = (3,3),padding = 'same', activation = 'relu') (x)
 
    # second decoder
    x = Conv2DTranspose(filters=256,kernel_size = (2,2), strides = 2) (x)   
    skip_features = tf.image.resize(e3, size = (x.shape[1], x.shape[2]))
    x = tf.keras.layers.Concatenate()([x, skip_features])
    x = Conv2D(filters=256,kernel_size = (3,3),padding = 'same', activation = 'relu') (x)
    x = Conv2D(filters=256,kernel_size = (3,3),padding = 'same', activation = 'relu') (x)
 
    # third decoder
    x = Conv2DTranspose(filters=128,kernel_size = (2,2), strides = 2) (x)   
    skip_features = tf.image.resize(e2, size = (x.shape[1], x.shape[2]))
    x = tf.keras.layers.Concatenate()([x, skip_features])
    x = Conv2D(filters=128,kernel_size = (3,3),padding = 'same', activation = 'relu') (x)
    x = Conv2D(filters=128,kernel_size = (3,3),padding = 'same', activation = 'relu') (x)
 
    # fourth decoder
    x = Conv2DTranspose(filters=64,kernel_size = (2,2), strides = 2) (x)   
    skip_features = tf.image.resize(e1, size = (x.shape[1], x.shape[2]))
    x = tf.keras.layers.Concatenate()([x, skip_features])
    x = Conv2D(filters=64,kernel_size = (3,3),padding = 'same', activation = 'relu') (x)
    x = Conv2D(filters=64,kernel_size = (3,3),padding = 'same', activation = 'relu') (x)
 
    # Output
    outputs = tf.keras.layers.Conv2D(1, 
                                    1, 
                                    padding = 'valid', 
                                    activation = 'sigmoid')(x)
     
    model = tf.keras.models.Model(inputs = inputs, 
                                 outputs = outputs, 
                                 name = 'U-Net')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss="binary_crossentropy",
                  metrics="MeanIoU")
    return  model