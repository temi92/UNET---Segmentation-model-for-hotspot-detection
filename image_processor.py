# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 15:34:58 2020

@author: odusi
"""

import tensorflow as tf
import random
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_and_preprocess_image(images, masks):
    image_list  =[]
    masks_list = []
    for img, mask in zip(images, masks):
        # Open, decode, resize, normalize
        image = tf.io.read_file(img)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (224, 224))
        image = image / 255.
        image_list.append(image.numpy())
        
        #mask
        mask = tf.io.read_file(mask)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, (224, 224))
        mask = mask / 255.
        masks_list.append(mask.numpy())
    
    return np.array(image_list), np.array(masks_list)

@tf.function
def load_and_preprocess_testImage(image):
    # Open, decode, resize, normalize
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = image / 255. 
    return image

def augment_images(X_train, Y_train, seed=0, batch_size=32, data_gen_args=dict(
        rotation_range=90.0,
        # width_shift_range=0.02,
        height_shift_range=0.02,
        shear_range=5,
        # zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="constant",
    )):
    
     # Train data, provide the same seed and keyword arguments to the fit and flow methods
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_train, augment=True, seed=seed)
    Y_datagen.fit(Y_train, augment=True, seed=seed)
    X_train_augmented = X_datagen.flow(
        X_train, batch_size=batch_size, shuffle=True, seed=seed
    )
    Y_train_augmented = Y_datagen.flow(
        Y_train, batch_size=batch_size, shuffle=True, seed=seed
    )

    train_generator = zip(X_train_augmented, Y_train_augmented)
    return train_generator

    
    





