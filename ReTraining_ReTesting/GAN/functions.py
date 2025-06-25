import os
import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras import layers
from PIL import Image
from classes import *

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

# Normalizing the images to [-1, 1]
def normalize(input_image):
  input_image = (input_image / 127.5) - 1
  return input_image

def VGG(target,gen_output_256):
  target_processed = keras.applications.vgg19.preprocess_input(target)
  gen_output_processed = keras.applications.vgg19.preprocess_input(gen_output_256)
  model = keras.applications.vgg19.VGG19(weights= 'imagenet', include_top= False)
  features_target = model(target_processed) 
  features_gen_output = model(gen_output_processed)
  return tf.reduce_mean(tf.keras.losses.MSE(features_target,features_gen_output))

def multiscale_loss(target,out256,out128,out64):
  list = [out256,out128,out64]
  loss = 0
  for image in list:
    if image.shape[-3:-1] != target.shape[-3:-1]:
        resized_target = tf.image.resize(target,image.shape[-3:-1])
        loss += tf.reduce_mean(tf.keras.losses.MSE(image,resized_target))
    else:
        loss += tf.reduce_mean(tf.keras.losses.MSE(image, target))
  return loss

#Define the Generator Loss
def generator_loss(disc_generated_output, gen_output_256, gen_output_128, gen_output_64, target, loss_object):
  # 1) Cast the images to float32 for ops like SSIM, VGG, etc.
  target          = tf.cast(target,          tf.float32)
  gen_output_256  = tf.cast(gen_output_256,  tf.float32)
  gen_output_128  = tf.cast(gen_output_128,  tf.float32)
  gen_output_64   = tf.cast(gen_output_64,   tf.float32)
  
  # 2) Compute losses, but cast their results to float32 as well
  # Adversarial
  adv = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
  adversarial_loss = tf.cast(adv, tf.float32)

  # Multi-scale
  ms = multiscale_loss(target, gen_output_256, gen_output_128, gen_output_64) / 3
  ms_loss = tf.cast(ms, tf.float32)

  # SSIM
  ssim_val = tf.image.ssim(
      img1=target,
      img2=gen_output_256,
      max_val=1.0,
      filter_size=11,
      filter_sigma=1.5,
      k1=0.01,
      k2=0.03
  )
  ssim = 1.0 - tf.reduce_mean(ssim_val)
  ssim = tf.cast(ssim, tf.float32)

  VGG
  vgg_val   = VGG(target, gen_output_256)
  vgg_loss  = tf.cast(vgg_val, tf.float32)

  # 3) Sum them up as float32
  total_gen_loss = adversarial_loss + ms_loss + ssim + vgg_loss

  return total_gen_loss, adversarial_loss, ms_loss, ssim, vgg_loss

#Define the discriminator loss
def discriminator_loss(disc_real_output, disc_generated_output, loss_object):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

#Define Generator Network
def Generator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = keras.Input(shape = (None,None,4))

    downsampling1,res1 = Down(64)(inputs)
    downsampling2,res2 = Down(128)(downsampling1)
    downsampling3,res3 = Down(256)(downsampling2)
    downsampling4,res4 = Down(512)(downsampling3)
    downsampling5,res5 = Down(1024)(downsampling4)

    x = CBL(2048)(downsampling5)
    x = x * ChannelAttention(2048)(x)
    x = x * SpatialAttention()(x)
    x = x + layers.Conv2D(2048,kernel_size = 3, strides = 1, padding = 'same', use_bias = False)(downsampling5)

    Upsampling1 = layers.Conv2DTranspose(filters = 1024, kernel_size = 2, strides = 2, padding = 'same')(x)
    combined_images1 = layers.Concatenate(axis = -1)([Upsampling1,res5])
    x = Up(1024)(combined_images1)
    x = x + Upsampling1

    Upsampling2 = layers.Conv2DTranspose(filters = 512, kernel_size = 2, strides = 2, padding = 'same')(x)
    combined_images2 = layers.Concatenate(axis = -1)([Upsampling2,res4])
    x = Up(512)(combined_images2)
    x = x + Upsampling2

    Upsampling3 = layers.Conv2DTranspose(filters = 256, kernel_size = 2, strides = 2, padding = 'same')(x)
    combined_images3 = layers.Concatenate(axis = -1)([Upsampling3,res3])
    x = Up(256)(combined_images3)
    out3 = x + Upsampling3

    Upsampling4 = layers.Conv2DTranspose(filters = 128, kernel_size = 2, strides = 2, padding = 'same')(out3)
    combined_images4 = layers.Concatenate(axis = -1)([Upsampling4,res2])
    x = Up(128)(combined_images4)
    out2 = x + Upsampling4

    Upsampling5 = layers.Conv2DTranspose(filters = 64, kernel_size = 2, strides = 2, padding = 'same')(out2)
    combined_images5 = layers.Concatenate(axis = -1)([Upsampling5,res1])
    x = Up(64)(combined_images5)
    out1 = x + Upsampling5

    output1 = layers.Conv2D(filters = 3, kernel_size = 3, strides = 1, padding = 'same', use_bias = False,activation= keras.activations.sigmoid)(out1)
    output2 = layers.Conv2D(filters = 3, kernel_size = 3, strides = 1, padding = 'same', use_bias = False,activation= keras.activations.sigmoid)(out2)
    output3 = layers.Conv2D(filters = 3, kernel_size = 3, strides = 1, padding = 'same', use_bias = False,activation= keras.activations.sigmoid)(out3)

    return keras.Model(inputs = inputs, outputs = [output1,output2,output3])

#Define the Discriminator
def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=(None, None, 3), name='input_image')

    x = layers.Conv2D(filters = 64, kernel_size=4, strides = 2, padding = 'same')(inp)
    x = layers.LeakyReLU(alpha = 0.2)(x)

    x = layers.Conv2D(filters = 128, kernel_size=4, strides = 2, padding = 'same',use_bias = False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha = 0.2)(x)

    x = layers.Conv2D(filters = 256, kernel_size=4, strides = 2, padding = 'same',use_bias = False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha = 0.2)(x)

    x = layers.Conv2D(filters = 512, kernel_size=4, strides = 2, padding = 'same',use_bias = False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha = 0.2)(x)

    x = layers.Conv2D(filters = 1, kernel_size=4, strides = 2, padding = 'same')(x)
    x = keras.activations.sigmoid(x)
    
    return keras.Model(inputs = inp,outputs = x)



