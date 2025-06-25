import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

from tensorflow import keras
from keras import layers
from PIL import Image
from functions import *


#Define the CBR (Convolutional, Batch Normalization and ReLU block)
class CBR(keras.Model):
    
    def __init__(self,filters):
        super().__init__()
        self.Convolution = layers.Conv2D(filters = filters, kernel_size = 3, strides = 1, padding = 'same', use_bias = False)
        self.BatchN = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self,inputs):
        x = self.Convolution(inputs)
        x = self.BatchN(x)
        return self.relu(x)    
    
#Define the Network
inputs = keras.Input(shape = (None,None,3))

CBR1 = CBR(64)(inputs)

CBR2 = CBR(64)(CBR1)

Downsampling1 = layers.MaxPooling2D(pool_size = (2,2), strides = 2)(CBR2)
CBR3 = CBR(128)(Downsampling1)

CBR4 = CBR(128)(CBR3)

Downsampling2 = layers.MaxPooling2D(pool_size = (2,2), strides = 2)(CBR4)
CBR5 = CBR(256)(Downsampling2)

CBR6 = CBR(256)(CBR5)

Downsampling3 = layers.MaxPooling2D(pool_size = (2,2), strides = 2)(CBR6)
CBR7 = CBR(512)(Downsampling3)

Upsampling1 = layers.Conv2DTranspose(filters = 256, kernel_size = 2, strides = 2, padding = 'same')(CBR7)
#Add the Skip connections
combined_images1 = layers.Concatenate(axis = -1)([Upsampling1,CBR6])
CBR8 = CBR(512)(combined_images1)

CBR9 = CBR(256)(CBR8)

Upsampling2 = layers.Conv2DTranspose(filters = 128, kernel_size = 2, strides = 2, padding = 'same')(CBR9)
combined_images2 = layers.Concatenate(axis = -1)([Upsampling2,CBR4])
CBR10 = CBR(256)(combined_images2)

CBR11 = CBR(128)(CBR10)

Upsampling3 = layers.Conv2DTranspose(filters = 64, kernel_size = 2, strides = 2, padding = 'same')(CBR11)
combined_images3 = layers.Concatenate(axis = -1)([Upsampling3,CBR2])
CBR12 = CBR(128)(combined_images3)

CBR13 = CBR(64)(CBR12)

output = layers.Conv2D(filters = 1, kernel_size = 3, strides = 1, padding = 'same')(CBR13)

model = keras.Model(inputs = inputs, outputs = output, name = 'Test')
model.summary()

#Load weights
model.load_weights('/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/Raindrop_Mask/5FoldCV/model1/attentionweights_val.weights.h5')

#Load the dataset
rain_images_a = np.asarray(load_dataset(top_dir = '/data/lady6758/Parmeet_Datasets/NewDatasets/A_Qian_Cam_A+B/Real/rain_images'))
rain_images_b = np.asarray(load_dataset(top_dir = '/data/lady6758/Parmeet_Datasets/NewDatasets/B_Quan/Real+Synthetic/rain_images'))
rain_images = np.append(rain_images_a,rain_images_b,axis = 0)
del(rain_images_a)
del(rain_images_b)

y_mask_a = np.asarray(load_dataset_g(top_dir = '/data/lady6758/Parmeet_Datasets/NewDatasets/A_Qian_Cam_A+B/Real/KwonMask'))
y_mask_b = np.asarray(load_dataset_g(top_dir = '/data/lady6758/Parmeet_Datasets/NewDatasets/B_Quan/Real+Synthetic/KwonMask'))
y_mask = np.append(y_mask_a,y_mask_b,axis = 0)
del(y_mask_a)
del(y_mask_b)

rain_images = np.float32(rain_images)
y_mask = np.float32(y_mask)

rain_images = rain_images/255
y_mask = y_mask/255

np.random.seed(12345)
np.random.shuffle(rain_images)

np.random.seed(12345)
np.random.shuffle(y_mask)

print(rain_images.shape)
print(y_mask.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((rain_images, y_mask))
SHUFFLE_BUFFER = 1000
BATCH_SIZE = 8

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER)
train_dataset = train_dataset.batch(BATCH_SIZE)


model.compile(
    loss=keras.losses.MeanSquaredError(), 
    optimizer = keras.optimizers.Adam(learning_rate = 0.0002 , beta_1 = 0.5, beta_2 = 0.999 ),
    metrics = ['accuracy'])


history = model.fit(train_dataset, epochs = 100)

np.save('/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/Raindrop_Mask/Fold1Retrained/history.npy',history.history)

model.save_weights(filepath = '/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/Raindrop_Mask/Fold1Retrained/attentionweights.weights.h5')
