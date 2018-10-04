import functools
import numpy as np
import tensorflow as tf
from keras import backend as K

def __semi_hard_triplet_loss(labels, embeddings, margin = 0.2):
    return tf.contrib.losses.metric_learning.triplet_semihard_loss(labels, embeddings, margin=margin)

def semi_hard_triplet_loss(margin):
    @functools.wraps(__semi_hard_triplet_loss)
    def loss(labels, embeddings):
        return __semi_hard_triplet_loss(labels, embeddings, margin)
    return loss