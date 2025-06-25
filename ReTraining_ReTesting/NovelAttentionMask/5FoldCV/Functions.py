import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow import keras
from keras import layers,activations
from Classes import *


def MaskModule():
    initializer = tf.random_normal_initializer(0., 0.02)
    filters=32
    inputs = keras.Input(shape = (None,None,3))
    test = inputs[:,:,:,0]
    test = tf.expand_dims(test,axis = -1)
    mask = tf.ones_like(test,dtype = tf.float32)/2

    test1 = tf.zeros_like(mask)
    test6 = tf.concat([inputs,inputs],axis = -1)
    test12 = tf.concat([test6,test6],axis = -1)
    test24 = tf.concat([test12,test12],axis = -1)
    test30 = tf.concat([test24,test6],axis = -1)
    test32 = tf.concat([tf.concat([test30,test1],axis = -1),test1],axis = -1)
    h = tf.zeros_like(test32,dtype = tf.float32)
    c = tf.zeros_like(test32,dtype = tf.float32)
    masks = []

    for a in range(5):
        x = layers.Concatenate(axis = -1)([inputs,mask])
        x = CBR(filters,initial = True)(x)
        resx = x
        x = activations.relu(CBR(filters)(x)+resx)
        resx = x
        x = activations.relu(CBR(filters)(x)+resx)
        resx = x
        x = activations.relu(CBR(filters)(x)+resx)
        resx = x
        x = activations.relu(CBR(filters)(x)+resx)
        resx = x
        x = activations.relu(CBR(filters)(x)+resx)
        
        x = layers.Concatenate(axis = -1)([x,h])
        i,f,g,o = LSTM(filters)(x)
        c = f*c + i*g
        h = o*activations.tanh(c)

        mask = layers.Conv2D(filters = 1, kernel_size= 3, strides = 1, padding = 'same')(h)
        mask = activations.relu(mask)
        masks.append(mask)
    return  keras.Model(inputs = inputs, outputs = masks)

#Load the training data
def load_dataset(top_dir = 'input_data'):
    images_dataset = []
    for name in sorted(os.listdir(top_dir)):
        img = np.array(Image.open(os.path.join(top_dir,name)))
        img = img[:,:,:3]
        images_dataset.append(img)
    return images_dataset

def load_dataset_g(top_dir = 'input_data'):
    images_dataset = []
    for name in sorted(os.listdir(top_dir)):
        img = np.array(Image.open(os.path.join(top_dir,name)))
        images_dataset.append(img)
    return images_dataset

def mse_loss(masks_array, target_mask):
    n = len(masks_array)
    mse = tf.keras.losses.MeanSquaredError()
    loss = 0
    theta = 0.5
    for i in range(n):
        loss = loss + (theta**(n-1-i)) * mse(masks_array[i],target_mask)
    return loss     

def mean_std(array):
    mean = np.mean(array)
    std = np.std(array)
    print(f'Mean: {mean}')
    print(f'STD: {std}')