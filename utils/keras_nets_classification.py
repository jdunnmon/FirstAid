import numpy as np
import keras

from keras import optimizers
from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.applications.resnet50 import ResNet50


def Keras_ResNet50_Net(layer,is_training,class_num,batch_size, keep_prob=1.0, name="Keras_ResNet50_Net",initial=None):
    """
    This is a 50-layer Keras ResNet wrapped for FirstAid
    INPUTS:
    - layer: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.
    - name: (str) the name of the network
    - initial: (str) 'imagenet' for pretrained, None for scratch
    """
    layer_shape = layer.get_shape().as_list()
    input_shape = (layer_shape[1],layer_shape[2],layer_shape[3])  

    model = ResNet50(include_top=False, weights=initial,
                 pooling='max', input_shape=input_shape)
    pred = Dense(class_num, activation='sigmoid')(model.layers[-1].output)
     
    return pred

