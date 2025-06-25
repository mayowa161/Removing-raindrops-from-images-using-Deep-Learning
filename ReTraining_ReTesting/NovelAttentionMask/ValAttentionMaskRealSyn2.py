import tensorflow as tf
import numpy as np
from tensorflow import keras
import pickle

from keras import layers
from Classes import *
from Functions import *
from PIL import Image
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim   
from sklearn.model_selection import KFold


### CREATE THE TRAINING SET ### 
rain_images_a = np.asarray(load_dataset(top_dir = '/data/lady6758/Parmeet_Datasets/NewDatasets/A_Qian_Cam_A+B/Real/rain_images'))
rain_images_b = np.asarray(load_dataset(top_dir = '/data/lady6758/Parmeet_Datasets/NewDatasets/B_Quan/Real+Synthetic/rain_images'))
rain_images = np.append(rain_images_a,rain_images_b,axis = 0)
del(rain_images_a)
del(rain_images_b)

mask_images_a = np.asarray(load_dataset_g(top_dir = '/data/lady6758/Parmeet_Datasets/NewDatasets/A_Qian_Cam_A+B/Real/AbsMask')) 
mask_images_b = np.asarray(load_dataset_g(top_dir = '/data/lady6758/Parmeet_Datasets/NewDatasets/B_Quan/Real+Synthetic/AbsMask')) 
mask_images = np.append(mask_images_a,mask_images_b,axis = 0)
mask_images = np.expand_dims(mask_images,axis = -1)

rain_images = np.float32(rain_images)
mask_images = np.float32(mask_images)

rain_images = rain_images/255
mask_images = mask_images/255

np.random.seed(12345)
np.random.shuffle(rain_images)

np.random.seed(12345)
np.random.shuffle(mask_images)

print(rain_images.shape)
print(mask_images.shape)

MaskModule = MaskModule()
MaskModule.save_weights(filepath = '/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/NovelAttentionMask/5FoldCV/maskmoduleinitialweights.weights.h5')

kf = KFold(n_splits = 5,shuffle=True, random_state=12345)

for i, (train_index, val_index) in enumerate(kf.split(rain_images)):   
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Validation:  index={val_index}")

    if i == 0:
        continue

    if i == 1:
        continue

    # if i == 2:
    #     continue

    # if i == 3:
    #     continue

    if i == 4:
        continue

    np.save('/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/NovelAttentionMask/5FoldCV/model'+str(i)+'/val.npy', np.asarray(val_index))
    MaskModule.load_weights(filepath = '/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/NovelAttentionMask/5FoldCV/maskmoduleinitialweights.weights.h5')
    
    rain_images_train = rain_images[train_index]
    mask_images_train = mask_images[train_index]

    rain_images_val = rain_images[val_index]
    mask_images_val = mask_images[val_index]

    train_dataset = tf.data.Dataset.from_tensor_slices((rain_images_train, mask_images_train))
    validation_dataset = tf.data.Dataset.from_tensor_slices((rain_images_val, mask_images_val))

    BATCH_SIZE = 8

    train_dataset = train_dataset.batch(BATCH_SIZE)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)

    print(train_dataset.cardinality().numpy())
    print(validation_dataset.cardinality().numpy())

    del(rain_images_train)
    del(mask_images_train)

    mask_optimizer = keras.optimizers.Adam(learning_rate = 0.0002 , beta_1 = 0.5, beta_2 = 0.999 )

    my_dict =  {'total_mse_loss':[],'val_mse_loss':[]}

    def train_step(rain_image, target_mask,training = True):
        with tf.GradientTape() as mask_tape:
            masks = MaskModule(rain_image, training=training)

            mseloss = mse_loss(masks,target_mask)

        if training == True:
            mask_gradients = mask_tape.gradient(mseloss,
                                                    MaskModule.trainable_variables)

            mask_optimizer.apply_gradients(zip(mask_gradients,
                                                    MaskModule.trainable_variables))
        return mseloss
    
    #Define the fit step

    def fit(train_dataset, validation_dataset,epochs):
        val_mse_best = 1000
        metric_names = ['total_mse_loss','val_mse_loss']
        number_of_batches = len([_ for _ in iter(train_dataset)])
        for epoch in range(epochs):
            val_mse_loss_array = []
            print("\nEpoch {}/{}".format(epoch+1,epochs))
            progBar = tf.keras.utils.Progbar(number_of_batches, stateful_metrics=metric_names)

            for idx,(input_image,target) in train_dataset.enumerate():

                total_mse_loss = train_step(input_image,target)

                values = [('total_mse_loss',np.asarray(total_mse_loss).item())]
                progBar.update(np.asarray(idx+1).item(), values=values)


            for (input_image,target) in validation_dataset:
                val_mse_loss = train_step(input_image,target, training = False)
                val_mse_loss_array.append(np.asarray(val_mse_loss).item())

            val_mse_loss = np.mean(np.asarray(val_mse_loss))


            values = [('val_mse_loss',np.asarray(val_mse_loss).item())]

            progBar.update(np.asarray(idx+1).item(), values=values, finalize = True)

            my_dict['total_mse_loss'].append(np.asarray(total_mse_loss).item())
            my_dict['val_mse_loss'].append(np.asarray(val_mse_loss).item())


            if epoch % 10 == 0:     
                MaskModule.save_weights('/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/NovelAttentionMask/5FoldCVV/model'+str(i)+'/maskmodule_weights.weights.h5')
                with open('/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/NovelAttentionMask/5FoldCV/model'+str(i)+'/loss.pkl', 'wb') as fp:
                    pickle.dump(my_dict, fp)
                    print('dictionary saved successfully to file')
                
            
            if val_mse_loss < val_mse_best:
                val_mse_best = val_mse_loss
                best_epoch = epoch + 1
                MaskModule.save_weights('/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/NovelAttentionMask/5FoldCV/model'+str(i)+'/maskmodule_weights_val.weights.h5')
                print('Saved Weights')

        # Print which epoch had the best val loss after training finishes
        print(f"Best performing model for fold {i} was saved at epoch {best_epoch}")

    #Train it
    fit(train_dataset, validation_dataset, epochs = 50)

    MaskModule.save_weights('/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/NovelAttentionMask/5FoldCV/model'+str(i)+'/maskmodule_weights.weights.h5')
    with open('/users/lady6758/Mayowa_4YP_Code/ReTraining_ReTesting/NovelAttentionMask/5FoldCV/model'+str(i)+'/loss.pkl', 'wb') as fp:
        pickle.dump(my_dict, fp)
        print('dictionary saved successfully to file')