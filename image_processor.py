# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 15:34:58 2020

@author: odusi
"""

import tensorflow as tf
import random

@tf.function
def load_and_preprocess_image(path_to_image):
    # Open, decode, resize, normalize
    image = tf.io.read_file(path_to_image)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.

    return image


@tf.function
def augment_image(image):
    # - Symmetries
    

    if random.random() > 0.905:
        image = tf.image.rot90(image, k=1)#k is the number of times image is rotated by 90 degerees.
    elif random.random() > 0.905:
        image = tf.image.rot90(image, k=2)
    elif random.random() > 0.905:
        image = tf.image.rot90(image, k=3)
    elif random.random() > 0.905:
        image = tf.image.flip_left_right(image)
    elif random.random() > 0.905:
        image = tf.image.flip_up_down(image)
    elif random.random() > 0.905:
        image = tf.image.flip_up_down(tf.image.rot90(image, k=1))
    elif random.random() > 0.905:
        image = tf.image.flip_left_right(tf.image.rot90(image, k=1))

    """
    if random.random() > 0.5:
        image = tf.image.random_brightness(image=image, max_delta=.1)

    if random.random() > 0.5:
        image = tf.image.random_contrast(image=image, lower=.9, upper=1.1)
    """
    return image
