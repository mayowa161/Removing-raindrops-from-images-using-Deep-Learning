import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# from image_similarity_measures.quality_metrics import ssim
from tensorflow import keras
from keras import layers
from PIL import Image
from functions import *
from sklearn.model_selection import KFold


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
# model.save_weights(filepath = '/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/Raindrop_Mask/5FoldCV/initial_weights.weights.h5')

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

kf = KFold(n_splits = 5,shuffle=True, random_state=12345)

for i, (train_index, val_index) in enumerate(kf.split(rain_images)):

    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Validation:  index={val_index}")

    if i == 0:
        continue
    if i == 1:
        continue
    if i == 2:
        continue
    if i == 3:
        continue
    # if i == 4:
    #     continue

    np.save('/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/Raindrop_Mask/5FoldCV/model'+str(i)+'/val.npy', np.asarray(val_index))

    model.load_weights(filepath = '//users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/Raindrop_Mask/5FoldCV/initial_weights.weights.h5')
    
    rain_images_train = rain_images[train_index]
    y_mask_train = y_mask[train_index]

    rain_images_val = rain_images[val_index]
    y_mask_val = y_mask[val_index]

    train_dataset = tf.data.Dataset.from_tensor_slices((rain_images_train, y_mask_train))
    validation_dataset = tf.data.Dataset.from_tensor_slices((rain_images_val, y_mask_val))

    BATCH_SIZE = 8

    train_dataset = train_dataset.batch(BATCH_SIZE)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)

    model.compile(
        loss=keras.losses.MeanSquaredError(), 
        optimizer = keras.optimizers.Adam(learning_rate = 0.0002 , beta_1 = 0.5, beta_2 = 0.999 ),
        metrics = ['accuracy'])

    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath = '/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/Raindrop_Mask/5FoldCV/model'+str(i)+'/attentionweights_val.weights.h5', 
                                                    save_weights_only = True, 
                                                    monitor = 'val_loss',
                                                    save_best_only = True)

    history = model.fit(train_dataset, epochs = 100, validation_data = validation_dataset, callbacks = model_checkpoint)

    # Print statement for best epoch
    best_epoch = np.argmin(history.history['val_loss']) + 1
    print(f"The best model for fold {i} was saved at epoch {best_epoch}")

    np.save('/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/Raindrop_Mask/5FoldCV/model'+str(i)+'/history.npy',history.history)
    
    model.save_weights(filepath = '/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/Raindrop_Mask/5FoldCV/model'+str(i)+'/attentionweights.weights.h5')


