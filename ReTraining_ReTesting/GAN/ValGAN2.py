import tensorflow as tf
import numpy as np 
import os
import time
import datetime
import pickle

from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from functions import *
from IPython import display
from sklearn.model_selection import KFold


### CREATE THE TRAINING SET ### 
#Load the dataset
rain_images_a = np.asarray(load_dataset(top_dir = '/data/lady6758/Parmeet_Datasets/NewDatasets/A_Qian_Cam_A+B/Real/rain_images'))
rain_images_b = np.asarray(load_dataset(top_dir = '/data/lady6758/Parmeet_Datasets/NewDatasets/B_Quan/Real+Synthetic/rain_images'))
rain_images = np.append(rain_images_a,rain_images_b,axis = 0)
del(rain_images_a)
del(rain_images_b)

clean_images_a = np.asarray(load_dataset(top_dir = '/data/lady6758/Parmeet_Datasets/NewDatasets/A_Qian_Cam_A+B/Real/clean_images'))
clean_images_b = np.asarray(load_dataset(top_dir = '/data/lady6758/Parmeet_Datasets/NewDatasets/B_Quan/Real+Synthetic/clean_images'))
clean_images = np.append(clean_images_a,clean_images_b,axis = 0)
del(clean_images_a)
del(clean_images_b)

mask_images = np.asarray(load_dataset_g(top_dir = '/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/GAN/ModelGeneratedKwonMask'))
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

generator = Generator()
discriminator = Discriminator()

# generator.save_weights(filepath = '/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/GAN/5FoldCV/generatorinitialweights.weights.h5')
# discriminator.save_weights(filepath = '/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/GAN/5FoldCV/discriminatorinitialweights.weights.h5')

kf = KFold(n_splits = 5,shuffle=True, random_state=12345)

for i, (train_index, val_index) in enumerate(kf.split(clean_images)):

  print(f"Fold {i}:")
  print(f"  Train: index={train_index}")
  print(f"  Validation:  index={val_index}")

  # if i == 0:
  #   continue
  # if i == 1:
  #   continue
  if i == 2:
    continue
  if i == 3:
    continue
  if i == 4:
    continue
  np.save('/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/GAN/5FoldCV/model'+str(i)+'/val.npy',np.asarray(val_index))

  generator.load_weights(filepath = '/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/GAN/5FoldCV/generatorinitialweights.weights.h5')
  discriminator.load_weights(filepath = '/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/GAN/5FoldCV/discriminatorinitialweights.weights.h5')

  #Create the datasets

  concat_images_train_set = concat_images[train_index]
  clean_images_train_set = clean_images[train_index]

  concat_images_val_set = concat_images[val_index]
  clean_images_val_set = clean_images[val_index]

  train_dataset = tf.data.Dataset.from_tensor_slices((concat_images_train_set, clean_images_train_set))
  train_batch = 8
  train_dataset = train_dataset.batch(train_batch)

  validation_dataset = tf.data.Dataset.from_tensor_slices((concat_images_val_set, clean_images_val_set))
  val_batch = 8
  validation_dataset = validation_dataset.batch(val_batch)

  print(train_dataset.cardinality().numpy())
  print(validation_dataset.cardinality().numpy())

  del(concat_images_train_set)
  del(clean_images_train_set)
  del(concat_images_val_set)
  del(clean_images_val_set)

  #Value for the mean loss
  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)

  #Optimizers
  generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5, beta_2= 0.999)
  discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5, beta_2= 0.999)


  #Define the training step

  my_dict =  {'total_gen_loss':[],'adversarial_loss':[],'ssim':[],'vgg':[],'ms_loss':[],'disc_loss':[],'val_gen_loss':[],'val_disc_loss':[]}

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

  def fit(train_dataset, validation_dataset,epochs):
      val_gen_best = 1000
      metric_names = ['total_gen_loss','adversarial_loss','ssim','vgg','ms_loss','disc_loss','val_gen_loss','val_disc_loss']
      number_of_batches = len([_ for _ in iter(train_dataset)])
      for epoch in range(epochs):
          val_gen_mean = []
          val_disc_mean = []
          print("\nEpoch {}/{}".format(epoch+1,epochs))
          progBar = tf.keras.utils.Progbar(number_of_batches, stateful_metrics=metric_names)

          for idx,(input_image,target) in train_dataset.enumerate():

              total_gen_loss,adversarial_loss,ssim,vgg,ms_loss,disc_loss = train_step(input_image,target)

              values = [('total_gen_loss',np.asarray(total_gen_loss).item()),('adversarial_loss',np.asarray(adversarial_loss).item()),('ssim',np.asarray(ssim).item()),('vgg',np.asarray(vgg).item()),
                        ('ms_loss',np.asarray(ms_loss).item()),('disc_loss',np.asarray(disc_loss).item())]
              progBar.update(np.asarray(idx+1).item(), values=values)

              #Test on validation data

          for (input_image,target) in validation_dataset:
              val_gen_loss,val_disc_loss = train_step(input_image,target, training = False)
              val_gen_mean.append(val_gen_loss)
              val_disc_mean.append(val_disc_loss)

          val_gen_loss = np.mean(np.asarray(val_gen_mean))
          val_disc_loss = np.mean(np.asarray(val_disc_mean))


          values = [('val_gen_loss',np.asarray(val_gen_loss).item()),('val_disc_loss',np.asarray(val_disc_loss).item())]

          progBar.update(np.asarray(idx+1).item(), values=values, finalize = True)

          my_dict['total_gen_loss'].append(np.asarray(total_gen_loss).item())
          my_dict['adversarial_loss'].append(np.asarray(adversarial_loss).item())
          my_dict['ssim'].append(np.asarray(ssim).item())
          my_dict['vgg'].append(np.asarray(vgg).item())
          my_dict['ms_loss'].append(np.asarray(ms_loss).item())
          my_dict['disc_loss'].append(np.asarray(disc_loss).item())
          my_dict['val_gen_loss'].append(np.asarray(val_gen_loss).item())
          my_dict['val_disc_loss'].append(np.asarray(val_disc_loss).item())

          if epoch % 10 == 0:     
            generator.save_weights('/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/GAN/5FoldCV/model'+str(i)+'/generator_weights.weights.h5')
            discriminator.save_weights('/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/GAN/5FoldCV/model'+str(i)+'/discriminator_weights.weights.h5')

            with open('/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/GAN/5FoldCV/model'+str(i)+'/loss.pkl', 'wb') as fp:
              pickle.dump(my_dict, fp)
              print('dictionary saved successfully to file')
            
          
          if val_gen_loss < val_gen_best:
            val_gen_best = val_gen_loss
            generator.save_weights('/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/GAN/5FoldCV/model'+str(i)+'/generatorweights_val.weights.h5')
            discriminator.save_weights('/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/GAN/5FoldCV/model'+str(i)+'/discriminatorweights_val.weights.h5')
            print(f"Saved Weights at epoch {epoch+1} for fold {i} with val_gen_loss={val_gen_loss}")


  #Train it
  fit(train_dataset, validation_dataset, epochs = 50)

  generator.save_weights('/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/GAN/5FoldCV/model'+str(i)+'/generator_weights.weights.h5')
  discriminator.save_weights('/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/GAN/5FoldCV/model'+str(i)+'/discriminator_weights.weights.h5')

  with open('/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/GAN/5FoldCV/model'+str(i)+'/loss.pkl', 'wb') as fp:
    pickle.dump(my_dict, fp)
    print('dictionary saved successfully to file')

