# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 20:19:25 2020

@author: odusi
"""
import tensorflow as tf
import argparse
from pathlib import Path
from image_processor import tf_dataset, normalize_mask
from utils.visualize_images import plot_orig_predMask
from loss import iou, dice_coef, dice_loss
import numpy as np

argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument('-d','--dataset',help='folder containing images',\
        required=True,
        default=None)
argparser.add_argument('-w','--weights', help='paths to weights file',\
        required=True,
        default=None)

args = argparser.parse_args()
def main():
    model = tf.keras.models.load_model(args.weights, custom_objects={"dice_loss":dice_loss, "dice_coef":dice_coef, "iou":iou})
    images = Path(args.dataset)
 
    img_paths = [str(path) for path in list(images.glob('*.JPG'))]
    test_dataset = tf_dataset(img_paths, batch=len(img_paths))
    test_dataset = test_dataset.as_numpy_iterator().next()
    
    predictions = model.predict(test_dataset)

    mask_predictions = []
    for m in range(predictions.shape[0]):
        
        mask = predictions[m]
        mask = normalize_mask(mask)
        mask_predictions.append(mask)
        
    mask_predictions = np.array(mask_predictions)
  
    plot_orig_predMask(test_dataset, predictions)    
    
if __name__=="__main__":
    main()
