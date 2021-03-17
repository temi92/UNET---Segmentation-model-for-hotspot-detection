# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 16:07:32 2020

@author: odusi
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import argparse
import os
from pathlib import Path
import glob
from image_processor import load_and_preprocess_image, augment_images
from utils.visualize_images import plot_images, plot_orig_predMask
from models.mobilenet_unet import unet_model
from loss import iou, dice_coef, dice_loss, bce_dice_loss
from datetime import datetime
import random
from sklearn.model_selection import train_test_split

argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument('-d','--dataset',help='folder containing images and masks',\
        required=True,
        default=None)
argparser.add_argument('-o', '--output', type=Path, help="folder where weights would be stored",\
                      required=True)
argparser.add_argument('-e','--epochs',type=int, help='number of epochs for training',\
        default=2)
argparser.add_argument('-lr','--learning_rate',type=float, default=0.01, help='number of epochs for training')
argparser.add_argument('-resume', default='', type=Path, help='path to latest checkpoint (default: none)')
       
argparser.add_argument('--batch_size',type=int, help="batch size" ,\
        default=16)       
    

args = argparser.parse_args()
batch_size = args.batch_size
epochs = args.epochs
output_folder = args.output
learning_rate = args.learning_rate
def main():
    if not output_folder.is_dir():
        os.mkdir(output_folder)
    
    weight_filename = os.path.join(output_folder, "weights-improvement-max-{epoch:02d}-{val_dice_coef:.3f}.hdf5")
    
    #tensorboard setup
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_dice_coef', patience=10, restore_best_weights=False, mode="max"),\
             tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=0,write_graph=True, write_images=True), 
             tf.keras.callbacks.ModelCheckpoint(weight_filename, verbose=1, monitor="val_dice_coef", mode="max", save_best_only=True)]      
        
    
        
        
    images = Path(args.dataset, "image")
    masks = Path(args.dataset, "mask")
    #find images and masks path.
    img_paths = sorted([str(path) for path in list(images.glob('*.jpg'))])
    masks_paths = sorted([str(path) for path in list(masks.glob('*.jpg'))])
      
  
    imgs_np, masks_np = load_and_preprocess_image(img_paths, masks_paths)
    
    x_train, x_val, y_train, y_val = train_test_split(imgs_np, masks_np, test_size=0.2, random_state=0)

    train_gen = augment_images(x_train, y_train, batch_size=batch_size)    

    
    model  = unet_model(finetune=False)
    
      
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate,
        decay_steps=1000000,
        decay_rate=0.5,
        staircase=True)
    
    
    #model.summary()

        
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=dice_loss, \
                 metrics=[dice_coef, iou])  
        
    #kick strat training by loading weights from previous training session.
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            model.load_weights(args.resume)

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
        

                    
    model.fit(train_gen, epochs=epochs,steps_per_epoch=100, validation_data=(x_val, y_val), callbacks=callbacks)
    
    
    

if __name__ == "__main__":
    main()