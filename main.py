# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 16:07:32 2020

@author: odusi
"""

import tensorflow as tf
import numpy as np
import argparse
from pathlib import Path
import glob
from image_processor import load_and_preprocess_image, augment_image
argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument('-d','--dataset',help='folder containing images and masks',\
        required=True,
        default=None)

args = argparser.parse_args()
batch_size = 16

def main():
    images = Path(args.dataset, "image")
    masks = Path(args.dataset, "mask")
    
    #find images and masks path.
    img_paths = [str(path) for path in list(images.glob('*.jpg'))]
    masks_paths = [str(path) for path in list(images.glob('*.png'))]

    #generate training and testing data set.
    train_slices = tf.data.Dataset.from_tensor_slices(img_paths)
    train_data = train_slices.map(load_and_preprocess_image).map(augment_image) \
                        .shuffle(buffer_size=10000).batch(batch_size)



if __name__ == "__main__":
    main()