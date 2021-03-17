# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 22:27:54 2020

@author: odusi
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from keras.models import load_model

from tensorflow.keras.layers import Dense, Input,Flatten, concatenate,Reshape, Conv2D, MaxPooling2D, Lambda,Activation,Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D, Conv2DTranspose, BatchNormalization, Dropout
# Convolution block with Transpose Convolution
def deconv_block(tensor, nfilters, size=3, padding='same', kernel_initializer = 'he_normal'):
    
    y = Conv2DTranspose(filters=nfilters, kernel_size=size, strides=2, padding = padding, kernel_initializer = kernel_initializer)(tensor)
    y = BatchNormalization()(y)
    y = Dropout(0.5)(y)
    y = Activation("relu")(y)  
    return y


# Convolution block with Upsampling+Conv2D
def deconv_block_rez(tensor, nfilters, size=3, padding='same', kernel_initializer = 'he_normal'):
    y = UpSampling2D(size = (2,2),interpolation='bilinear')(tensor)
    y = Conv2D(filters=nfilters, kernel_size=(size,size), padding = 'same', kernel_initializer = kernel_initializer)(y)
    y = BatchNormalization()(y)
    #y = Dropout(0.5)(y)
    y = Activation("relu")(y)
    return y


def unet_model(finetune=False):
   
    # Encoder/Feature extractor
    mnv2=tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224, 224, 3),alpha=0.5, include_top=False, weights='imagenet')
    

    if (finetune):
      for layer in mnv2.layers[:-10]:
        layer.trainable = True
        
  
    
    x = mnv2.layers[-4].output

    # Decoder
    x = deconv_block_rez(x, 512)
    x = concatenate([x, mnv2.get_layer('block_13_expand_relu').output], axis = 3)
    x = deconv_block_rez(x, 256)
    
    x = concatenate([x, mnv2.get_layer('block_6_expand_relu').output], axis = 3)
                
    x = deconv_block_rez(x, 128)
    x = concatenate([x, mnv2.get_layer('block_3_expand_relu').output], axis = 3)
    
    x = deconv_block_rez(x, 64)
    x = concatenate([x, mnv2.get_layer('block_1_expand_relu').output], axis = 3)
                

    x = UpSampling2D(size = (2,2),interpolation='bilinear')(x)
    x = Conv2D(filters=32, kernel_size=3, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
   
    x = Conv2DTranspose(1, (1,1), padding='same')(x)
    x = Activation('sigmoid', name="op")(x)
    model = Model(inputs=mnv2.input, outputs=x)    
    return model

unet_model()
