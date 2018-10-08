from __future__ import print_function

import os
import yaml
import argparse
import numpy as np
import tensorflow as tf
import keras.optimizers as optimizers

from keras import losses
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from loss import *
from data import DataLoader
from model import get_model

def get_loss_function(func):
    return {
        'large-margin-cosine-loss': large_margin_cos_loss(config["train"]),
        'intra-enhanced-triplet-loss': intra_enhanced_triplet_loss(config["train"], config["data"]),
        'semi-hard-triplet-loss': semi_hard_triplet_loss(config["train"]["alpha"]),
        'categorical-crossentropy': losses.categorical_crossentropy,
    }.get(func, losses.categorical_crossentropy)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Model Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="config.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.load(file)

    paths = config["paths"]
    train = config["train"]
    data  = config["data"]

    if not os.path.exists(paths["save"]): os.makedirs(paths["save"])

    with open(os.path.join(paths["save"], config["run-title"] + ".yaml"), 'w') as outfile:
        yaml.dump(config, outfile)

    dataloader = DataLoader(data, config["train"]["loss"]=="categorical-crossentropy")
    dataloader.load()

    input_shape = (data["imsize"], data["imsize"], data["imchannel"])
    model = get_model(input_shape, config, top=config["train"]["loss"]=="categorical-crossentropy")

    if train["resume"]: 
        model.load_weights(paths["load"], by_name=True)

    loss_func = get_loss_function(config["train"]["loss"])
    optim = getattr(optimizers, train["optim"])(train["lr"])
    model.compile(loss=loss_func, optimizer=optim, metrics=[])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=train["lr_reduce_factor"], patience=train["patience"], min_lr=train["min_lr"])
    checkpoint = ModelCheckpoint(os.path.join(paths["save"],"model.{epoch:02d}-{val_loss:.4f}.h5"), monitor='val_loss', save_best_only=True, mode='min')
    tensorboard = TensorBoard(log_dir=os.path.join('./Graph',config["run-title"]), histogram_freq=0, write_graph=True, write_images=True)

    model.fit(dataloader.X_train, dataloader.y_train,
        epochs=train["epochs"],
        batch_size=train["batch-size"],
        verbose=1,
        shuffle=True,
        validation_split=data["val_split"],
        callbacks=[checkpoint, reduce_lr, tensorboard]
    )