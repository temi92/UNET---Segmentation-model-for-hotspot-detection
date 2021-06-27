# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 13:40:40 2021

@author: odusi
"""

import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import os
import cv2
import argparse
import glob
import shutil



input_dir = "train_test_data_1/image"
output_dir = "train_test_data_1/image_aug"
mask_dir = "train_test_data_1/mask"
output_mask_dir = "train_test_data_1/mask_aug"


seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Flipud(0.5),
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    #iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (1.8, 1.8), "y": (1.8, 1.8)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-5, 5),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order

#copy original files to destination folder ..
for filename in glob.glob(os.path.join(input_dir, "*.jpg")):
    shutil.copy(filename, output_dir)
    
    
for filename in glob.glob(os.path.join(mask_dir, "*.png")):
    base = os.path.splitext(os.path.basename(filename))[0]
    temp_img = cv2.imread(filename)
    cv2.imwrite(os.path.join(output_mask_dir, base+".jpg"), temp_img)
 
    #shutil.copy(filename, output_mask_dir)

#write augmentated samples to the destination folder
for  i in range(50):
    for filename in glob.glob(os.path.join(input_dir, "*.jpg")):
        
        im = cv2.imread(filename)
        im = cv2.resize(im, (640, 360))
        
        
        base = os.path.splitext(os.path.basename(filename))[0]
        im_mask = cv2.imread(os.path.join(mask_dir, base+ ".png"))
        im_mask = cv2.resize(im_mask, (640, 360))
        segmap = SegmentationMapsOnImage(im_mask, shape=im.shape)
       
        #import sys
        #sys.exit(1)
    
        img_aug, seg_aug = seq(image=im, segmentation_maps=segmap)
    
        cv2.imwrite(os.path.join(output_dir, base+ str(i)+ "_aug"+ ".jpg"),img_aug)
        cv2.imwrite(os.path.join(output_mask_dir, base+ str(i)+ "_aug"+ ".jpg"),seg_aug.get_arr())


