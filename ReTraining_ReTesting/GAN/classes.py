import tensorflow as tf
from tensorflow import keras
from keras import layers

class ChannelAttention(keras.layers.Layer):
    def __init__(self,channels,ratio = 16):
        super().__init__()
        self.avg_pool = layers.GlobalAvgPool2D(keepdims=True)
        self.max_pool = layers.GlobalMaxPool2D(keepdims = True)
        
        self.fc = keras.Sequential([
            layers.Conv2D(filters = channels//ratio, kernel_size=1, use_bias= False),
            layers.ReLU(),
            layers.Conv2D(filters = channels, kernel_size = 1, use_bias = False)
        ])
    def call(self,inputs):
        avg_out = self.fc(self.avg_pool(inputs))
        max_out = self.fc(self.max_pool(inputs))
    
        out = avg_out + max_out
        return keras.activations.sigmoid(out)

class SpatialAttention(keras.layers.Layer):
    def __init__(self,kernel_size = 7):
        super().__init__()
        self.fc = keras.Sequential([
            layers.ZeroPadding2D(kernel_size//2),
            layers.Conv2D(filters = 1, kernel_size = kernel_size, use_bias = False,activation = keras.activations.sigmoid),
        ])
    
    def call(self,inputs):
        avg_out = tf.reduce_mean(inputs,axis = -1,keepdims=True)
        max_out = tf.reduce_max(inputs,axis = -1,keepdims=True)
        x = layers.Concatenate(axis = -1)([avg_out,max_out])
        return self.fc(x)
    
class CBL(keras.layers.Layer):
    def __init__(self,filters,relu = True):
        super().__init__()
        self.Convolution = layers.Conv2D(filters = filters,kernel_size= 3, strides = 1, padding = 'same', use_bias=False)
        self.BatchNorm = layers.BatchNormalization()
        self.LReLU = layers.LeakyReLU(alpha = 0.2)
        self.relu = relu

    def call(self,inputs, training = None):
        x = self.Convolution(inputs)
        x = self.BatchNorm(x)
        if self.relu == True:
            return self.LReLU(x)
        return x

class Down(keras.layers.Layer):
    def __init__(self,filters):
        super().__init__()
        self.CBL1 = CBL(filters, relu = True)
        self.CBL2 = CBL(filters, relu = False)
        self.ChannelAttention = ChannelAttention(channels=filters)
        self.SpatialAttention = SpatialAttention()
        self.Conv2D = layers.Conv2D(filters,kernel_size = 3, strides = 1, padding = 'same', use_bias = False)
        self.relu = layers.ReLU()
        self.MaxPool = layers.MaxPooling2D(pool_size = (2,2), strides = 2)
    
    def call(self,inputs):
        x = self.CBL1(inputs)
        x = self.CBL2(x)
        x = x * self.ChannelAttention(x)
        x = x * self.SpatialAttention(x)
        res = x + self.Conv2D(inputs)
        res = self.relu(res)
        Downsampling = self.MaxPool(res)
        return Downsampling,res

class Up(keras.layers.Layer):
    def __init__(self,filters):
        super().__init__()
        self.CBL1 = CBL(filters*2)
        self.CBL2 = CBL(filters, relu = False)
        self.ChannelAttention = ChannelAttention(channels=filters)
        self.SpatialAttention = SpatialAttention()
        self.relu = layers.ReLU()

    def call(self,inputs):
        x = self.CBL1(inputs)
        x = self.CBL2(x)
        x = x * self.ChannelAttention(x)
        x = x * self.SpatialAttention(x)
        return self.relu(x)

