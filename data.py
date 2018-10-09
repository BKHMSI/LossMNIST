from __future__ import print_function

import os
import yaml
import argparse
import numpy as np 

from keras.utils import np_utils
from keras.datasets import mnist

class DataLoader(object):
    def __init__(self, config, one_hot = False):
        self.config = config
        self.one_hot = one_hot

    def load(self):
        (X_data, self.y_data), (X_test, self.y_test) = mnist.load_data()

        self.input_shape = (-1, self.config["data"]["imsize"], self.config["data"]["imsize"], self.config["data"]["imchannel"])
        self.X_data = np.reshape(X_data, self.input_shape)
        self.X_test  = np.reshape(X_test, self.input_shape)

        if self.one_hot:
            self.y_data = np_utils.to_categorical(self.y_data, self.config["data"]["num_classes"])
            self.y_test = np_utils.to_categorical(self.y_test, self.config["data"]["num_classes"])

        self.num_train = int(self.y_data.shape[0] * (1-self.config["data"]["val_split"]))
        self.num_val   = int(self.y_data.shape[0] * (self.config["data"]["val_split"]))
        self.num_test  = self.y_test.shape[0]


    def preprocess(self, data):
        data = data.astype('float32')
        # data = data - self.mean  
        # data = data / self.std
        return data / 255.

    def order_data_triplet_loss(self):
        data = {}
        samples_per_id = self.config["data"]["samples_per_id"]
        for label in range(self.config["data"]["num_classes"]):
            mask = self.y_data==label
            data[label] = [i for i, x in enumerate(mask) if x]
            if len(data[label]) < samples_per_id:
                data[label].extend(np.random.choice(data[label], samples_per_id - len(data[label]), replace=False))
            data[label] = data[label][:samples_per_id]

        k_batch = self.config["train"]["k_batch"]
        X_data, y_data = [], []
        for i in range(samples_per_id // k_batch):
            for label in data:
                X_data.extend(self.X_data[data[label][i*k_batch:(i+1)*k_batch]])
                y_data += [label] * k_batch

        self.X_data = np.array(X_data)
        self.y_data = np.array(y_data)

    def split_data(self):
        self.X_train = self.X_data[:self.num_train]
        self.y_train = self.y_data[:self.num_train]

        self.X_val = self.X_data[self.num_train:]
        self.y_val = self.y_data[self.num_train:]

        self.mean = np.mean(self.X_train, axis=0) 
        self.std  = np.std(self.X_train, axis=0)
        self.std  = (self.std==0) * 1e-16 + self.std 

        self.X_train = self.preprocess(self.X_train)
        self.X_val   = self.preprocess(self.X_val)
        self.X_test  = self.preprocess(self.X_test)
        del self.X_data, self.y_data
                    
    def get_random_batch(self, k = 100):
        X_batch, y_batch = [], []
        for label in range(self.config["num_classes"]):
            X_mask = self.X_test[self.y_test==label]
            X_batch.extend(np.array([X_mask[np.random.choice(len(X_mask), k, replace=False)]]) if k <= len(X_mask) and k >= 0 else X_mask)
            y_batch += [label] * k if k <= len(X_mask) and k >= 0 else [label] * len(X_mask)
        X_batch = np.reshape(X_batch, self.input_shape)
        return X_batch, np.array(y_batch)


class DataGenerator(object):
    def __init__(self, config):
        self.shuffle = config["train"]["shuffle"]
        self.batch_size = config["train"]["batch-size"]

    def generate(self, X, y):
        ''' Generates batches of samples '''
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(len(y))
            # Generate batches
            batches = np.arange(len(indexes)//self.batch_size)
            np.random.shuffle(batches)

            for batch in batches:
                # Find list of ids
                batch_indecies = indexes[batch*self.batch_size:(batch+1)*self.batch_size]
                yield X[batch_indecies], y[batch_indecies]

    def __get_exploration_order(self, data_size):
        ''' Generates order of exploration '''
        idxs = np.arange(data_size)
        if self.shuffle == True:
            np.random.shuffle(idxs)
        return idxs   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="config.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.load(file)

    dataloader = DataLoader(config)
    dataloader.load()
    dataloader.order_data_triplet_loss()
