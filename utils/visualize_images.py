# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:10:27 2020

@author: odusi
"""

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
    
    
def plot_orig_predMask(orig_imgs, pred_imgs):
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(orig_imgs.shape[0], 2, figsize=(20, 20), squeeze=False)
    axes[0,0].set_title("original", fontsize=15)
    axes[0,1].set_title("pred mask", fontsize=15)
    for m in range(0, orig_imgs.shape[0]):
        axes[m,0].imshow(orig_imgs[m])
        axes[m, 0].set_axis_off()
        axes[m,1].imshow(pred_imgs[m])
        axes[m, 1].set_axis_off()
    plt.tight_layout()
    plt.show()
        
