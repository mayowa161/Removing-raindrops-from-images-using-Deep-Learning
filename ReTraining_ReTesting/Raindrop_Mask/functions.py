import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

from tensorflow import keras
from keras import layers
from PIL import Image

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

#Normalize the images between 0 and 1
def normalize(list):
    for i in range(len(list)):
        list[i] = (list[i]-np.min(list[i]))/(np.max(list[i])-np.min(list[i]))
    return list

#Expand the dims of each image
def expand_dims(list):
    for i in range(len(list)):
        list[i] = np.expand_dims(list[i], axis = 0)
    return list

#Create validation data
def val_data(list1,list2,size):
    for i in range(size):
        list2[i] = list1[i]
        list1.pop(i)
    return list1, list2  

#Reshape the images to 720x480
def shape_resize(img):
    if img.shape[0] != 480 or img.shape[1] != 720:
        #print ('before alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
        #align to four
        a_row = int(img.shape[0]/480)*480
        a_col = int(img.shape[1]/720)*720
        img = img[0:a_row, 0:a_col]
        #print ('after alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    return img

#Generate the masks
def mask_func(image):
    b,g,r = cv2.split(image)
    I = 0.299*r+0.587*g+0.114*b
    return I

def rescale(mask):
    mask_scaled = 255*(mask-np.min(mask))/(np.max(mask)-np.min(mask))
    #mask_scaled = mask_scaled.astype(int)
    return mask_scaled

#Without section - Performs cropping of the images
def crop_image(clean_image,degraded_image, i_columns, i_rows, out_length):
    #Define an empty array
    cropped_image_clean = np.empty((256,256,3), dtype = int)
    cropped_image_degraded = np.empty((256,256,3), dtype = int)
    #section = np.empty((i_rows,i_columns),dtype = int)
    #Pick a random starting point in the image
    x_start = np.random.randint(0,i_columns - out_length)
    y_start = np.random.randint(0, i_rows - out_length)
    #Split the image into its RGB
    b_c,g_c,r_c = cv2.split(clean_image)
    b_d,g_d,r_d = cv2.split(degraded_image)
    for j in range(256):
        for i in range(256):
            cropped_image_clean[(j,i,0)] = b_c[(y_start+j,x_start+i)]
            cropped_image_clean[(j,i,1)] = g_c[(y_start+j,x_start+i)]
            cropped_image_clean[(j,i,2)] = r_c[(y_start+j,x_start+i)]

            cropped_image_degraded[(j,i,0)] = b_d[(y_start+j,x_start+i)]
            cropped_image_degraded[(j,i,1)] = g_d[(y_start+j,x_start+i)]
            cropped_image_degraded[(j,i,2)] = r_d[(y_start+j,x_start+i)]
            #section[(y_start+j,x_start+i)] = 255
    return cropped_image_clean, cropped_image_degraded

#Flip the images
def flip_image(clean_image,degraded_image):
    #Generate a number between 0 and 2 with equal probability
    number = np.random.randint(0,3)
    if number == 0:
        #Flip Vertically
        return np.flipud(clean_image),np.flipud(degraded_image)
    elif number == 1:
        #Flip Horizontally
        return np.fliplr(clean_image),np.fliplr(degraded_image)
    else:
        #return unflipped image 
        return clean_image,degraded_image