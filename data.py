import numpy as np 

from keras.utils import np_utils
from keras.datasets import mnist

class DataLoader(object):
    def __init__(self, config):
        self.config = config

    def load(self):
        (X_train, self.y_train), (X_test, self.y_test) = mnist.load_data()

        self.input_shape = (-1, self.config["imsize"], self.config["imsize"], self.config["imchannel"])
        X_train = np.reshape(X_train, self.input_shape)
        X_test  = np.reshape(X_test, self.input_shape)

        self.mean = np.mean(X_train, axis=0) 
        self.std = np.std(X_train, axis=0)
        self.std = (self.std==0) * 1e-16 + self.std 

        self.X_train = self.preprocess(X_train)
        self.X_test  = self.preprocess(X_test)

        self.Y_train = np_utils.to_categorical(self.y_train, self.config["num_classes"])
        self.Y_test = np_utils.to_categorical(self.y_test, self.config["num_classes"])

    def preprocess(self, data):
        data = data.astype('float32')
        data = data - self.mean  
        data = data / self.std
        return data

    def get_random_batch(self, k = 100):
        X_batch, y_batch = [], []
        for label in range(self.config["num_classes"]):
            mask = self.y_test==label
            X_batch += [self.X_test[mask][np.random.choice(np.sum(mask), k, replace=False)]]
            y_batch += [label] * k
        X_batch = np.reshape(np.array(X_batch), self.input_shape)
        return X_batch, np.array(y_batch)

