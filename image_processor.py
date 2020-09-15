# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 15:34:58 2020

@author: odusi
"""

import tensorflow as tf
import random

@tf.function
def load_and_preprocess_image(image, mask):
    # Open, decode, resize, normalize
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.
    
    #mask
    mask = tf.io.read_file(mask)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (224, 224))
    mask = mask / 255.
    
    return image, mask

@tf.function
def load_and_preprocess_testImage(image):
    # Open, decode, resize, normalize
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = image / 255. 
    return image


@tf.function
def augment_image(image, mask):
    # - Symmetries
    

    if random.random() > 0.905:
        image = tf.image.rot90(image, k=1)#k is the number of times image is rotated by 90 degerees.
        mask = tf.image.rot90(mask, k=1)
    elif random.random() > 0.905:
        image = tf.image.rot90(image, k=2)
        mask = tf.image.rot90(mask, k=2)

    elif random.random() > 0.905:
        image = tf.image.rot90(image, k=3)
        mask = tf.image.rot90(mask, k=3)

    elif random.random() > 0.905:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    elif random.random() > 0.905:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_left_right(mask)

    elif random.random() > 0.905:
        image = tf.image.flip_up_down(tf.image.rot90(image, k=1))
        mask = tf.image.flip_up_down(tf.image.rot90(mask, k=1))

    elif random.random() > 0.905:
        image = tf.image.flip_left_right(tf.image.rot90(image, k=1))
        mask = tf.image.flip_left_right(tf.image.rot90(mask, k=1))

    """
    if random.random() > 0.5:
        image = tf.image.random_brightness(image=image, max_delta=.1)

    if random.random() > 0.5:
        image = tf.image.random_contrast(image=image, lower=.9, upper=1.1)
    """
    return image, mask




