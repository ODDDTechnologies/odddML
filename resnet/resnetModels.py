from tensorflow.keras import models 
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from odddML.resnet.resnet_layers import * 
import tensorflow as tf 





class ResNet50(tf.keras.Model):
    """
    # Arguments
        num_classes: number of classes of your problem
    #### Usage: Use it as a keras model, just compile it with your loss function (probably categorical_crossentropy) and fit it on the data.  
    """
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.layer0 = PreRes()
        self.layer1 = build_res_block_2(filter_num=64, blocks=3)
        self.layer2 = build_res_block_2(filter_num=128, blocks=4, stride=2)
        self.layer3 = build_res_block_2(filter_num=256, blocks=6, stride=2)
        self.layer4 = build_res_block_2(filter_num=512, blocks=3, stride=2)
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(units=num_classes, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None):
        x = self.layer0(inputs, training=training) 
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

