# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 16:07:32 2020

@author: odusi
"""

import tensorflow as tf
import argparse
import os
from pathlib import Path
from datetime import datetime
from image_processor import load_dataset, tf_dataset
from models.mobilenet_unet import unet_model
from models.resnet_unet import resnet50_unet
from loss import iou, dice_coef, dice_loss

argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument('-d','--dataset',help='folder containing images and masks',\
        required=True,
        default=None)
argparser.add_argument('-o', '--output', type=Path, help="folder where weights would be stored",\
                      required=True)
argparser.add_argument('-e','--epochs',type=int, help='number of epochs for training',\
        default=30)
argparser.add_argument('-lr','--learning_rate',type=float, default=0.01, help='number of epochs for training')
argparser.add_argument('-resume', default='', type=Path, help='path to latest checkpoint (default: none)')
       
argparser.add_argument('--batch_size',type=int, help="batch size" ,\
        default=16)       
    

args = argparser.parse_args()
batch_size = args.batch_size
epochs = args.epochs
output_folder = args.output
learning_rate = args.learning_rate


IMG_SIZE = (224, 224, 3)
def main():
    if not output_folder.is_dir():
        os.mkdir(output_folder)
    
    weight_filename = os.path.join(output_folder, "weights-improvement-max-{epoch:02d}-{val_dice_coef:.3f}.hdf5")
    
    #tensorboard setup
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_dice_coef', patience=30, restore_best_weights=False, mode="max"),\
             tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=0,write_graph=True, write_images=True), 
             tf.keras.callbacks.ModelCheckpoint(weight_filename, verbose=1, monitor="val_dice_coef", mode="max", save_best_only=True)]      
        
    
    
    (train_x, train_y), (test_x, test_y) = load_dataset(args.dataset)
 

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    val_dataset = tf_dataset(test_x, test_y, batch=batch_size)

    #model  = unet_model(finetune=False)
    model = resnet50_unet(input_shape=IMG_SIZE)
    #model.summary()
    #TODO use Reduce learning rate on plateau functionality../
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=dice_loss, \
                 metrics=[dice_coef, iou])  
        
    #kick start training by loading weights from previous training session.

    
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        #model.load_model(args.resume)
        model = tf.keras.models.load_model(args.resume, custom_objects={"dice_loss":dice_loss, "dice_coef":dice_coef, "iou":iou})
       
        #model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=dice_loss, \
                 #metrics=[dice_coef, iou])  
                
                
        #model.set_weights(args.resume)
        #print(tf.keras.backend.get_value(model.optimizer.lr))

        #tf.keras.backend.set_value(model.optimizer.lr, 0.1)
      

    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        
    train_steps = len(train_x)//batch_size
    if len(train_x) % batch_size != 0:
        train_steps += 1

    test_steps = len(test_x)//batch_size
    if len(test_x) % batch_size != 0:
        test_steps += 1

                    
    model.fit(train_dataset, epochs=epochs,steps_per_epoch=train_steps, validation_data=val_dataset, validation_steps=test_steps, callbacks=callbacks)


if __name__ == "__main__":
    main()