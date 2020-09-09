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
from models.mobilenet_unet import unet_model
from loss import DiceLoss, iou, dice_coef
from datetime import datetime
import random
#tensoboard setup..
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


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


def display(display_list):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 15))

    title = ['Input Image', "Predicted Mask"]

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()
    
def plot_images(orig_imgs, mask_imgs, pred_imgs):
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(orig_imgs.shape[0], 3, figsize=(20, 20), squeeze=False)
    axes[0,0].set_title("original", fontsize=15)
    axes[0,1].set_title("original mask", fontsize=15)
    axes[0,2].set_title("predicted mask", fontsize=15)
    
    for m in range(0, orig_imgs.shape[0]):
        axes[m,0].imshow(orig_imgs[m])
        axes[m, 0].set_axis_off()
        axes[m,1].imshow(mask_imgs[m])
        axes[m, 1].set_axis_off()
        axes[m,2].imshow(pred_imgs[m])
        axes[m, 2].set_axis_off()

    plt.tight_layout()
    plt.show()
    
    
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


    #generate testing data
    val_dataset = tf.data.Dataset.from_tensor_slices(test_imgs)
    val_dataset = val_dataset.map(load_and_preprocess_testImage).batch(batch_size)

    val_dataset_mask = tf.data.Dataset.from_tensor_slices(test_masks)
    val_dataset_mask = val_dataset_mask.map(load_and_preprocess_testImage).batch(batch_size)

    model  = unet_model()
    
      
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01/10,
        decay_steps=1000000,
        decay_rate=0.5,
        staircase=True)
    
    
    #model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=DiceLoss(), \
                  metrics=[dice_coef, iou])

    
                        
    #model.fit(train_data, epochs=50,callbacks=[tensorboard_callback])
    #model.save_weights("model.h5")
    model.load_weights("model.h5")
    #model.fit(train_data, epochs=5,callbacks=[tensorboard_callback])
    
  
    
    pred_mask = model.predict(val_dataset)


    orig_imgs = np.zeros((SAMPLE_IMAGES, 224, 224, 3), dtype=np.float32)
    groundtruth_mask = np.zeros((SAMPLE_IMAGES, 224,224,3), dtype=np.float32)
    
    
    assert len(list(val_dataset.as_numpy_iterator())) == 1
    assert len(list(val_dataset_mask.as_numpy_iterator())) == 1

    k = list(val_dataset.as_numpy_iterator())[0]
    print ("********")
    print (k.shape)
    
    orig_imgs = np.copy(list(val_dataset.as_numpy_iterator())[0])

    groundtruth_mask = np.copy(list(val_dataset_mask.as_numpy_iterator())[0])
    
    print (orig_imgs.shape)
    
    #display([masks[0], output[0]])
    plot_images(orig_imgs, groundtruth_mask, pred_mask)
    
        
    

if __name__ == "__main__":
    main()