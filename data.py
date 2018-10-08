import numpy as np 

from keras.utils import np_utils
from keras.datasets import mnist

class DataLoader(object):
    def __init__(self, config, one_hot = False):
        self.config = config
        self.one_hot = one_hot

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

        if self.one_hot:
            self.y_train = np_utils.to_categorical(self.y_train, self.config["num_classes"])
            self.y_test = np_utils.to_categorical(self.y_test, self.config["num_classes"])

        self.num_train = int(self.y_train.shape[0] * (1-self.config["val_split"]))
        self.num_val   = int(self.y_train.shape[0] * (self.config["val_split"]))
        self.num_test  = self.y_test.shape[0]


    def preprocess(self, data):
        data = data.astype('float32')
        data = data - self.mean  
        data = data / self.std
        return data

    def get_random_batch(self, k = 100):
        X_batch, y_batch = [], []
        for label in range(self.config["num_classes"]):
            X_mask = self.X_test[self.y_test==label]
            X_batch.extend(np.array([X_mask[np.random.choice(len(X_mask), k, replace=False)]]) if k <= len(X_mask) and k >= 0 else X_mask)
            y_batch += [label] * k if k <= len(X_mask) and k >= 0 else [label] * len(X_mask)
        X_batch = np.reshape(X_batch, self.input_shape)
        return X_batch, np.array(y_batch)

