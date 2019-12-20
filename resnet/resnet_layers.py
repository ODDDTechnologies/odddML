from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, Multiply, Activation
from tensorflow.keras.initializers import glorot_uniform
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import os



class PreRes(tf.keras.layers.Layer):
    """
    # Arguments
        input: the tensor you want to pass through the layer
    #### Usage: Use it as a keras layer, PreRes has all of the attributes of the Layer API.
    #### Description: Conv2D --> BatchNormalization --> RelU --> MaxPool2D  
    """ 
    def __init__(self, **kwargs):
        super(PreRes, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activate = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                              strides=2)
    def call(self, input_tensor, training=None): 
        x = self.conv1(input_tensor)
        x = self.batch_norm(x)
        x = self.activate(x)
        x = self.max_pool(x)
        return x


class ResidualBlock(tf.keras.layers.Layer):
    """
    # Arguments
        input: the tensor you want to pass through the layer
    #### Usage: Use it as a keras layer, ResidualBlock has all of the attributes of the Layer API.
    #### Description: Implementation of the ResidualBlock of ResNet in keras
    """ 
    def __init__(self, filter_num, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=stride, padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=stride, padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(1, 1), strides=stride))
        self.downsample.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training=None):
        x = self.downsample(inputs)
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = tf.nn.relu(tf.keras.layers.add([shortcut, x]))

        return x


class ConvoBlock(tf.keras.layers.Layer):
    """
    # Arguments
        input: the tensor you want to pass through the layer
    #### Usage: Use it as a keras layer, ConvoBlock has all of the attributes of the Layer API.
    #### Description: Implementation of the ConvoBlock or Bottleneck of ResNet in tensorflow subclass API
    """
    def __init__(self, filter_num, stride=1):
        super(ConvoBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(1, 1), strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=stride, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=filter_num * 4, kernel_size=(1, 1), strides=1, padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num * 4, kernel_size=(1, 1), strides=stride))
        self.downsample.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training=None):
        shortcut = self.downsample(inputs)
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(tf.keras.layers.add([shortcut, x]))

        return x

def build_res_block_1(filter_num, blocks, stride=1):
    """
    # Arguments
        input: number of filters of the Residual blocks
        blocks: how many blocks you want to build
        stride: the stride of the layers inside the blocks
    ##### Returns: the built residual blocks
    #### Usage: Use it to create many ResidualBlocks
    """
    res_block = tf.keras.Sequential()
    res_block.add(ResidualBlock(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(ResidualBlock(filter_num, stride=1))

    return res_block


def build_res_block_2(filter_num, blocks, stride=1):
    """
    # Arguments
        input: number of filters of the Convo blocks
        blocks: how many blocks you want to build
        stride: the stride of the layers inside the blocks
    ##### Returns: the built convo blocks (bottlenecks)
    #### Usage: Use it to create many ConvoBlocks

    """
    res_block = tf.keras.Sequential()
    res_block.add(ConvoBlock(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(ConvoBlock(filter_num, stride=1))

    return res_block