import tensorflow as tf
from tensorflow import keras
from keras import layers,activations

class CBR(keras.layers.Layer):
    def __init__(self,filters, initial = False):
        super().__init__()
        self.Convolution = layers.Conv2D(filters = filters,kernel_size= 3, strides = 1, padding = 'same')
        self.BatchNorm = layers.BatchNormalization()
        self.initial = initial

    def call(self,inputs):
        x = self.Convolution(inputs)
        x = self.BatchNorm(x)
        x = activations.relu(x)
        if self.initial == True:
            return x
        x = self.Convolution(x)
        x = self.BatchNorm(x)
        return activations.relu(x)
    
class LSTM(keras.layers.Layer):
    def __init__(self,filters):
        super().__init__()
        self.Conv2D = layers.Conv2D(filters = filters, kernel_size= 3, strides = 1, padding = 'same')

    def call(self,inputs):
        i = activations.sigmoid(self.Conv2D(inputs))
        f = activations.sigmoid(self.Conv2D(inputs))
        g = activations.tanh(self.Conv2D(inputs))
        o = activations.sigmoid(self.Conv2D(inputs))
        return i,f,g,o
    


