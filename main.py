# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 16:07:32 2020

@author: odusi
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import argparse
from pathlib import Path
import glob
from image_processor import load_and_preprocess_image, load_and_preprocess_testImage, augment_image
from utils.visualize_images import plot_images, plot_orig_predMask
from models.mobilenet_unet import unet_model
from loss import iou, dice_coef, dice_loss
from datetime import datetime
import random
#tensoboard setup..
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
#model_filename = "model.h5"

model_filename="weights-improvement-max-{epoch:02d}-{dice_coef:.2f}.hdf5"


#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='dice_coef', patience=10, restore_best_weights=False, mode="max"),\
             tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=0,write_graph=True, write_images=True), 
             tf.keras.callbacks.ModelCheckpoint(model_filename, verbose=1, monitor="dice_coef", mode="max", save_best_only=True)]

argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument('-d','--dataset',help='folder containing images and masks',\
        required=True,
        default=None)

argparser.add_argument('--input_height',type=int, help="input height" ,\
        default=224)    
    
argparser.add_argument('--input_width',type=int, help="input width" ,\
        default=224)

argparser.add_argument('--batch_size',type=int, help="batch size" ,\
        default=16)       
    

args = argparser.parse_args()
batch_size = args.batch_size
IMG_SIZE = (args.input_width, args.input_height)
SAMPLE_IMAGES = 5


   
    
def sample(img_path, mask_path):
    images = []
    masks = []
    for i in range(SAMPLE_IMAGES):
        index = int(random.uniform(0, len(img_path)))     
        images.append(img_path[index])
        masks.append(mask_path[index])
    return images, masks
    
def main():
    images = Path(args.dataset, "image")
    masks = Path(args.dataset, "mask")
    #find images and masks path.
    img_paths = [str(path) for path in list(images.glob('*.jpg'))]
    masks_paths = [str(path) for path in list(masks.glob('*.png'))]
    
    test_imgs, test_masks = sample(img_paths, masks_paths)
  
    

    #generate training data
    train_slices = tf.data.Dataset.from_tensor_slices((img_paths, masks_paths))
    
    train_data = train_slices.map(load_and_preprocess_image).map(augment_image) \
                        .shuffle(buffer_size=10000).batch(batch_size)


    """
    
    #generate testing data
    val_dataset = tf.data.Dataset.from_tensor_slices(test_imgs)
    val_dataset = val_dataset.map(load_and_preprocess_testImage).batch(batch_size)

    val_dataset_mask = tf.data.Dataset.from_tensor_slices(test_masks)
    val_dataset_mask = val_dataset_mask.map(load_and_preprocess_testImage).batch(batch_size)

    """


    model  = unet_model()
    
      
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01,
        decay_steps=1000000,
        decay_rate=0.5,
        staircase=True)
    
    
    #model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=dice_loss, \
                  metrics=[dice_coef, iou])
                    
    model.fit(train_data.repeat(), epochs=80,steps_per_epoch=100, callbacks=callbacks)

    
    
    
    """
    model.load_weights("weights-improvement-max-22-0.26.hdf5")
    
  
    
    
    pred_mask = model.predict(val_dataset)

    
    assert len(list(val_dataset.as_numpy_iterator())) == 1
    assert len(list(val_dataset_mask.as_numpy_iterator())) == 1

    
    orig_imgs = list(val_dataset.as_numpy_iterator())[0]
    groundtruth_mask = list(val_dataset_mask.as_numpy_iterator())[0]
    
    plot_images(orig_imgs, groundtruth_mask, pred_mask)
    """
    
    

if __name__ == "__main__":
    main()