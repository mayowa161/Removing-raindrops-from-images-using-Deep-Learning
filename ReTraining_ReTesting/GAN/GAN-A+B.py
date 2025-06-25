import tensorflow as tf
import numpy
import os
import time
import datetime
import pickle

from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from functions import *
from IPython import display

currentfolder = '(A+B)(Real+Synthetic)_Cam_A+B'
datasetfoldera = 'A_Qian_Cam_A+B/Real'
datasetfolderb = 'B_Quan/Real+Synthetic'

### CREATE THE TRAINING SET ### 
#Load the dataset
rain_images_a = np.asarray(load_dataset(top_dir = '/data/scat8633/NewDatasets/'+datasetfoldera+'/rain_images'))
rain_images_b = np.asarray(load_dataset(top_dir = '/data/scat8633/NewDatasets/'+datasetfolderb+'/rain_images'))
rain_images = np.append(rain_images_a,rain_images_b,axis = 0)
del(rain_images_a)
del(rain_images_b)

clean_images_a = np.asarray(load_dataset(top_dir = '/data/scat8633/NewDatasets/'+datasetfoldera+'/clean_images'))
clean_images_b = np.asarray(load_dataset(top_dir = '/data/scat8633/NewDatasets/'+datasetfolderb+'/clean_images'))
clean_images = np.append(clean_images_a,clean_images_b,axis = 0)
del(clean_images_a)
del(clean_images_b)

mask_images = np.asarray(load_dataset_g(top_dir = '/users/scat8633/code/GAN/'+currentfolder+'/ModelGeneratedKwonMaskFinal/'))
mask_images = np.expand_dims(mask_images,axis = -1)

#Concatenate the rain image and masks to create the training data 
concat_images = np.concatenate((rain_images,mask_images), axis = -1)

del(rain_images)
del(mask_images)

#Convert to float32
concat_images = np.float16(concat_images)
clean_images = np.float16(clean_images)

#Normalise to [0,1]
concat_images = concat_images/255
clean_images = clean_images/255

np.random.seed(12345)
np.random.shuffle(concat_images)

np.random.seed(12345)
np.random.shuffle(clean_images)

#Check the shape
print(concat_images.shape)
print(clean_images.shape)

#Create the datasets
train_dataset = tf.data.Dataset.from_tensor_slices((concat_images,clean_images))

train_shuffle = 1000
train_batch = 8
train_dataset = train_dataset.shuffle(train_shuffle)
train_dataset = train_dataset.batch(train_batch)

generator = Generator()
discriminator = Discriminator()

#generator.load_weights(filepath = '/users/scat8633/code/GAN/'+currentfolder+'/5FoldCV/model4/generatorweights_val.h5')
#discriminator.load_weights(filepath = '/users/scat8633/code/GAN/'+currentfolder+'/5FoldCV/model4/discriminatorweights_val.h5')

#Value for the mean loss
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)

#Optimizers
generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5, beta_2= 0.999)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5, beta_2= 0.999)

#Define the training step

my_dict =  {'total_gen_loss':[],'adversarial_loss':[],'ssim':[],'vgg':[],'ms_loss':[],'disc_loss':[]}

def train_step(input_image, target,training = True):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    out1, out2, out3 = generator(input_image, training=training)

    disc_real_output = discriminator(target, training=training)
    disc_generated_output = discriminator(out1, training=training)

    total_gen_loss, adversarial_loss, ms_loss, ssim, vgg = generator_loss(disc_generated_output, out1, out2, out3, target, loss_object)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output, loss_object)

  if training == True:
    generator_gradients = gen_tape.gradient(total_gen_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                            discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                            discriminator.trainable_variables))

  if training == True:
    return total_gen_loss,adversarial_loss,ssim,vgg,ms_loss,disc_loss
  else:
    return total_gen_loss, disc_loss


#Define the fit step

def fit(train_dataset,epochs):
    metric_names = ['total_gen_loss','adversarial_loss','ssim','vgg','ms_loss','disc_loss']
    number_of_batches = len([_ for _ in iter(train_dataset)])
    for epoch in range(epochs):
        print("\nEpoch {}/{}".format(epoch+1,epochs))
        progBar = tf.keras.utils.Progbar(number_of_batches, stateful_metrics=metric_names)

        for idx,(input_image,target) in train_dataset.enumerate():

            total_gen_loss,adversarial_loss,ssim,vgg,ms_loss,disc_loss = train_step(input_image,target)

            values = [('total_gen_loss',np.asarray(total_gen_loss).item()),('adversarial_loss',np.asarray(adversarial_loss).item()),('ssim',np.asarray(ssim).item()),('vgg',np.asarray(vgg).item()),
                      ('ms_loss',np.asarray(ms_loss).item()),('disc_loss',np.asarray(disc_loss).item())]
            progBar.update(np.asarray(idx+1).item(), values=values)

        progBar.update(np.asarray(idx+1).item(), values=values, finalize = True)

        if np.asarray(disc_loss).item() < 1:

          with open('/users/scat8633/code/GAN/'+currentfolder+'/Final/loss.pkl', 'wb') as fp:
              pickle.dump(my_dict, fp)
              print('dictionary saved successfully to file')

          print('Ended Training as Discriminator Won - Saved Previous Epoch Weights')
          break

        my_dict['total_gen_loss'].append(np.asarray(total_gen_loss).item())
        my_dict['adversarial_loss'].append(np.asarray(adversarial_loss).item())
        my_dict['ssim'].append(np.asarray(ssim).item())
        my_dict['vgg'].append(np.asarray(vgg).item())
        my_dict['ms_loss'].append(np.asarray(ms_loss).item())
        my_dict['disc_loss'].append(np.asarray(disc_loss).item())

        generator.save_weights('/users/scat8633/code/GAN/'+currentfolder+'/Final/generator_weights.h5')
        discriminator.save_weights('/users/scat8633/code/GAN/'+currentfolder+'/Final/discriminator_weights.h5')

        if epoch % 10 == 0:     
           with open('/users/scat8633/code/GAN/'+currentfolder+'/Final/loss.pkl', 'wb') as fp:
              pickle.dump(my_dict, fp)
              print('dictionary saved successfully to file')
              


#Train it
fit(train_dataset, epochs = 29)

generator.save_weights('/users/scat8633/code/GAN/'+currentfolder+'/Final/generator_weights.h5')
discriminator.save_weights('/users/scat8633/code/GAN/'+currentfolder+'/Final/discriminator_weights.h5')


with open('/users/scat8633/code/GAN/'+currentfolder+'/Final/loss.pkl', 'wb') as fp:
  pickle.dump(my_dict, fp)
  print('dictionary saved successfully to file')

