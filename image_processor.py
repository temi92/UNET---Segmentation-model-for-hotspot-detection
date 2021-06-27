# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 00:24:01 2021

@author: odusi
"""

import os
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf


IMG_SIZE = (224, 224)
def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, IMG_SIZE)
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, IMG_SIZE)
    x = x/255.0
    x = x > 0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def normalize_mask(mask, threshold=0.5):
    mask[mask > threshold] = 1
    mask[mask <= threshold] = 0
    return mask

def load_dataset(dataset_path):
    images = sorted(glob(os.path.join(dataset_path, "image/*")))
    masks = sorted(glob(os.path.join(dataset_path, "mask/*")))
   
  
    train_x, test_x = train_test_split(images, test_size=0.2, random_state=42)
    train_y, test_y = train_test_split(masks, test_size=0.2, random_state=42)

    return (train_x, train_y), (test_x, test_y)

def preprocess(image_path, mask_path):
    def f(image_path, mask_path):
        image_path = image_path.decode()
        mask_path = mask_path.decode()

        x = read_image(image_path)
        y = read_mask(mask_path)

        return x, y

    image, mask = tf.numpy_function(f, [image_path, mask_path], [tf.float32, tf.float32])
    image.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    mask.set_shape([IMG_SIZE[0], IMG_SIZE[1], 1])

    return image, mask

def preprocessTestImgs(image_path):
    def f(image_path):
        image_path = image_path.decode()
        x = read_image(image_path)
        return x
    image = tf.numpy_function(f, [image_path], [tf.float32])[0]
 
    
    image.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    return image

def tf_dataset(images, masks=None, batch=8):
    if masks is None:
        #used for evalutation script to show prediction from model.
        dataset = tf.data.Dataset.from_tensor_slices(images)
        dataset = dataset.map(preprocessTestImgs)
        dataset = dataset.batch(batch)
        dataset = dataset.prefetch(2)
        return dataset
        
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset




if __name__ == "__main__":
    dataset_path = "train_test_data_5"
    (train_x, train_y), (test_x, test_y) = load_dataset(dataset_path)
 
    """
    train_dataset = tf_dataset(train_x, train_y, batch=8)

    for images, masks in train_dataset:
        print(images.shape, masks.shape)
    """
    dataset = tf_dataset(train_x)
    
    #img = dataset.as_numpy_iterator().next()
    #print (img.shape)