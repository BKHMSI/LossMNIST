from __future__ import print_function

import os
import yaml
import argparse

import tensorflow as tf
import keras.backend as K 

from keras.models import Model
from keras.regularizers import l2
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Input, GlobalAveragePooling2D, LeakyReLU, SeparableConv2D, BatchNormalization, Add
from keras.layers.core import Dense, Dropout, Flatten, Lambda

def get_model(input_shape, config, top = True):
    input_img = Input(input_shape)
    num_classes = config["data"]["num_classes"]

    def __body(input_img):
        x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_img)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        embedding = Dense(128, activation='relu')(x)
        return embedding

    def __head(embedding):
        x   = Dropout(0.5)(embedding)
        out = Dense(num_classes, activation='softmax')(x)
        return out

    x = __body(input_img)
    if config["train"]["loss"] in ["triplet-softmax"] and top:
        y = __head(x)
        model = Model(inputs=input_img, outputs=[x, y])
    else:
        if top: x = __head(x)
        model = Model(inputs=input_img, outputs=x)
    return model

def simple_resnet(input_shape):

    repetitions = [1,1,1]
    def add_common_layers(x):
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x

    def residual_block(x, mul = 1, is_shortcut = False):
        shortcut = x

        x = SeparableConv2D(16 * mul, 1, padding="same", kernel_regularizer=l2(1e-4))(x)
        x = add_common_layers(x)

        x = SeparableConv2D(16 * mul, 3, padding="same", kernel_regularizer=l2(1e-4))(x)
        x = add_common_layers(x)

        x = SeparableConv2D(32 * mul, 1, padding="same", kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)

        if is_shortcut:
            shortcut = SeparableConv2D(32 * mul, 1, padding='same', kernel_regularizer=l2(1e-4))(shortcut)
            shortcut = BatchNormalization()(shortcut)

        x = Add()([x, shortcut])
        x = LeakyReLU()(x)

        return x

    input_img = Input(input_shape)
    x = Conv2D(32, 7, strides=2, padding="same", kernel_regularizer=l2(1e-4))(input_img)
    x = LeakyReLU()(x)
    x = MaxPooling2D(2, strides=2, padding="same")(x)

    for i, r in enumerate(repetitions):
        for j in range(r):
            x = residual_block(x, mul = 2**i, is_shortcut = (j==0))
        if i < len(repetitions) - 1:
            x = MaxPooling2D(2, strides=2, padding="same")(x)
        else:
            x = AveragePooling2D(2, strides=7)(x)
            
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=input_img, outputs=x)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="config.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.load(file)

    data  = config["data"]
    input_shape = (data["imsize"], data["imsize"], data["imchannel"])

    model = get_model(input_shape, config, top = True)
    model.summary()
    print("Parameter: {}".format(model.count_params()))