from __future__ import print_function

import os
import yaml
import argparse
import numpy as np
import tensorflow as tf
import keras.optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from data import DataLoader
from model import get_model

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

    dataloader = DataLoader(data)
    dataloader.load()

    input_shape = (data["imsize"], data["imsize"], data["imchannel"])
    model = get_model(input_shape, config)

    if train["resume"]: 
        model.load_weights(paths["load"], by_name=True)

    optim = getattr(keras.optimizers, train["optim"])(train["lr"])
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=train["lr_reduce_factor"], patience=train["patience"], min_lr=train["min_lr"])
    checkpoint = ModelCheckpoint(os.path.join(paths["save"],"model.{epoch:02d}-{val_loss:.4f}.h5"), monitor='val_loss', save_best_only=True, mode='min')

    model.fit(dataloader.X_train, dataloader.Y_train,
        batch_size=train["batch-size"],
        epochs=train["epochs"],
        verbose=1,
        shuffle=True,
        validation_split=data["val_split"],
        callbacks=[checkpoint, reduce_lr]
    )