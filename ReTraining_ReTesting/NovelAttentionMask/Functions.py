import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow import keras
from keras import layers,activations
from Classes import *


import tensorflow as tf
from tensorflow import keras
from keras import layers, activations

def MaskModule():
    initializer = tf.random_normal_initializer(0., 0.02)
    filters = 32
    
    # Keras symbolic input
    inputs = keras.Input(shape=(None, None, 3))
    
    # Wrap slicing, expand_dims, ones_like, zeros_like, concat, etc. in Lambda layers
    test = layers.Lambda(lambda x: tf.expand_dims(x[:, :, :, 0], axis=-1))(inputs)
    mask = layers.Lambda(lambda t: tf.ones_like(t, dtype=tf.float16) / 2)(test)

    test1 = layers.Lambda(lambda t: tf.zeros_like(t))(mask)
    test6 = layers.Lambda(lambda t: tf.concat([t, t], axis=-1))(inputs)
    test12 = layers.Lambda(lambda t: tf.concat([t, t], axis=-1))(test6)
    test24 = layers.Lambda(lambda t: tf.concat([t, t], axis=-1))(test12)
    
    test30 = layers.Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))([test24, test6])
    test31 = layers.Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))([test30, test1])
    test32 = layers.Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))([test31, test1])
    
    h = layers.Lambda(lambda t: tf.zeros_like(t, dtype=tf.float16))(test32)
    c = layers.Lambda(lambda t: tf.zeros_like(t, dtype=tf.float16))(test32)
    
    masks = []

    for a in range(5):
        x = layers.Concatenate(axis=-1)([inputs, mask])
        x = CBR(filters, initial=True)(x)  # Assuming CBR is your own layer
        resx = x
        x = activations.relu(CBR(filters)(x) + resx)
        resx = x
        x = activations.relu(CBR(filters)(x) + resx)
        resx = x
        x = activations.relu(CBR(filters)(x) + resx)
        resx = x
        x = activations.relu(CBR(filters)(x) + resx)
        resx = x
        x = activations.relu(CBR(filters)(x) + resx)
        
        x = layers.Concatenate(axis=-1)([x, h])
        i, f, g, o = LSTM(filters)(x)  # Assuming LSTM is your own layer
        c = f * c + i * g
        h = o * activations.tanh(c)

        mask = layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='same')(h)
        mask = activations.relu(mask)
        masks.append(mask)

    return keras.Model(inputs=inputs, outputs=masks)


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