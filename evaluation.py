# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 20:19:25 2020

@author: odusi
"""
import tensorflow as tf
import argparse
from pathlib import Path
from models.mobilenet_unet import unet_model
from image_processor import load_and_preprocess_testImage
from utils.visualize_images import plot_orig_predMask

model_filename = "weights-improvement-max-60-0.26.hdf5"

argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument('-d','--dataset',help='folder containing images',\
        required=True,
        default=None)

args = argparser.parse_args()
batch_size=16


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
    

def main():
    model = unet_model()
    model.load_weights(model_filename)
    images = Path(args.dataset)
    
    img_paths = [str(path) for path in list(images.glob('*.jpg'))]
    test_dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    test_dataset = test_dataset.map(load_and_preprocess_testImage).batch(batch_size)
    
    
    predictions = model.predict(test_dataset)
    assert len(list(test_dataset.as_numpy_iterator())) == 1
    
    test_dataset = list(test_dataset.as_numpy_iterator())[0] 

    
    plot_orig_predMask(test_dataset, predictions)    
    
if __name__=="__main__":
    main()
