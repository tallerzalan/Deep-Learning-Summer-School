#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 09:06:17 2022

@author: dtrzc
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 21:42:04 2022

@author: dtrzc
"""

import os, os.path, shutil
import numpy as np
import pandas as pd 
import random
import cv2
import matplotlib.pyplot as plt

import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, LeakyReLU, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf

seed = 2137
np.random.seed(seed)
tf.random.set_seed(seed)

full_folder_path = '/Users/dtrzc/Library/CloudStorage/OneDrive-Personal/0_MASTERS/3_Deep_Learning/data'

# images1 = [f for f in os.listdir(full_folder_path) if os.path.isfile(os.path.join(full_folder_path, f))]
# images1.pop(415)
# images1.remove('385_normal.jpg')
# images1.remove('385_pneumonia.jpg')
# print(len(images1))

# for image in images1:
#   folder_name = image.split("_")[1].split(".")[0]
#   new_path = os.path.join(full_folder_path, folder_name)
#   if not os.path.exists(new_path):
#     os.makedirs(new_path)

#   old_image_path = os.path.join(full_folder_path, image)
#   new_image_path = os.path.join(new_path, image)
#   shutil.move(old_image_path, new_image_path)

batch_size = 16 

#The dimension of the images we are going to define is 500x500 

img_height = 500
img_width = 500

# Create Image Data Generator for Train Set
image_gen = ImageDataGenerator(
                                rescale = 1./255,
                                rotation_range=10, 
                                zoom_range = 0.1, 
                                shear_range = 0.1       
                               )
# Create Image Data Generator for Test/Validation Set
# test_data_gen = ImageDataGenerator(rescale = 1./255)

train = image_gen.flow_from_directory(
      full_folder_path,
      target_size=(img_height, img_width),
      color_mode='grayscale',
      class_mode='binary',
      batch_size=batch_size
      )

# =============================================================================
# model
# =============================================================================

model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(activation = 'relu', units = 128))
model.add(Dense(activation = 'relu', units = 64))
model.add(Dense(activation = 'sigmoid', units = 1))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.summary()

# =============================================================================
# Callbacks
# =============================================================================

learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)
checkpoint = ModelCheckpoint(filepath='group_11_full_data_model.h5', save_best_only=True)
callbacks_list = [learning_rate_reduction, checkpoint]

# =============================================================================
# fit
# =============================================================================
hist = model.fit(train,epochs=18, callbacks=callbacks_list)

pd.DataFrame(hist.history).plot()