#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 10:23:55 2023

@author: julien
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.utils import load_img
import os
from display import print_random_figures_from_folder,create_narray, create_narray_resize

from SkinMaskModel import Unet_model

model = Unet_model()


input_shape = (None,None,None,3)
model.build(input_shape)
model.summary()


# Define the paths to your image and mask directories
train_image_dir = "../../Datasets/SkinDataset/Train/skin-images/"
train_mask_dir = "../../Datasets/SkinDataset/Train/masks/"

validation_image_dir = "../../Datasets/SkinDataset/Validation/skin-images/"
validation_mask_dir = "../../Datasets/SkinDataset/Validation/masks/"



# display some images

print_random_figures_from_folder(train_image_dir,train_mask_dir)

# Create Train data
(X_train, Y_train) = create_narray_resize(train_image_dir,train_mask_dir,257)
(X_validation, Y_validation) = create_narray_resize(validation_image_dir,validation_mask_dir,257)

"""
# Save the array to a file
np.save('X_train.npy', X_train)
np.save('Y_train.npy', Y_train)
np.save('X_validation.npy', X_validation)
np.save('Y_validation.npy', Y_validation)


# Load the array from the file
X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')
X_validation = np.load('X_validation.npy')
Y_validation = np.load('Y_validation.npy')
"""

# Run and save
checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto', save_freq="epoch")

model = tf.keras.models.load_model("best_model.hdf5")


batch_size = 32
tot_images = len(X_train)
num_steps_per_epoch = tot_images // batch_size
epochs = 5
history = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation),
                    steps_per_epoch=num_steps_per_epoch,
                    epochs=epochs,
                    verbose=1,
                    callbacks=checkpoint
)


# Saving the model
model.save('Skin.keras')

# Plot training error.
print('\nPlot of training error over 20 epochs:')
plt.title('Training error')
plt.ylabel('Cost')
plt.xlabel('epoch')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train loss', 'val loss'], loc='upper right')
plt.show()


"""
# Load a sample image and mask
img = load_img(train_image_dir+"2378.jpg")
mask = load_img(train_mask_dir+"2378.pbm",color_mode="grayscale")
print(mask)
plt.imshow((img))

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Convert the images to arrays
img_array = tf.keras.utils.img_to_array(img)
mask_array = tf.keras.utils.img_to_array(mask)

# Reshape the mask to binary (0 or 1)
mask_array = (mask_array > 0).astype(np.uint8)

# Expand dimensions to add a batch dimension
img_array = np.expand_dims(img_array, axis=0)
mask_array = np.expand_dims(mask_array, axis=0)

# Create a data generator for training
seed = 1  # Set a random seed for reproducibility
image_generator = datagen.flow(img_array, seed=seed)
mask_generator = datagen.flow(mask_array, seed=seed)

# Combine the image and mask generators to yield pairs
train_generator = zip(image_generator, mask_generator)

# Specify the batch size and number of training steps per epoch
batch_size = 32
tot_images = len(os.listdir(train_mask_dir))
num_steps_per_epoch = tot_images // batch_size
epochs = 10

history = model.fit(
    train_generator,
    steps_per_epoch=num_steps_per_epoch,
    epochs=epochs,
    # validation_data=validation_generator,  # Uncomment for validation
    # validation_steps=len(X_val) // batch_size,
    verbose=1
)

"""


