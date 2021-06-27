# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 17:28:21 2021

@author: odusi
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.layers import Dense, Input,Flatten, Concatenate, Reshape, Conv2D, MaxPooling2D, Lambda,Activation,Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D, Conv2DTranspose, BatchNormalization, Dropout


def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x) 
    return x

def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x= conv_block(x, num_filters)
    return x


def resnet50_unet(input_shape=(512, 512, 3)):
    inputs = Input(input_shape, name="input")
    
    resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
    
    
    """Encoder """
    
    s1= resnet50.get_layer("input").output #512x512x3
    s2= resnet50.get_layer("conv1_relu").output #256 x256x64
    s3= resnet50.get_layer("conv2_block3_out").output #128x128x256
    s4= resnet50.get_layer("conv3_block4_out").output # 64x64x512
    
    """Bottleneck"""
    
    b1 = resnet50.get_layer("conv4_block6_out").output #32x32x1024
    
    """Decoder"""
    
    d1 = decoder_block(b1,s4, 512) #64
    d2 = decoder_block(d1,s3, 258) #128
    d3 = decoder_block(d2,s2, 128) #256
    d4 = decoder_block(d3,s1, 64) #512

    """Outputs"""
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    
    model = Model(inputs, outputs)
    return model
  
if __name__=="__main__":
    input_shape = (512, 512, 3)
    model = resnet50_unet(input_shape)
    model.summary()
