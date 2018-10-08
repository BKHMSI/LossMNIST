import functools
import numpy as np
import tensorflow as tf
from keras import backend as K

def __anchor_center_loss(embeddings, margin, batch_size = 240, k = 4):
    """Computes the anchor-center loss

    Minimizes intra-class distances. Assumes embeddings are ordered 
    such that every k samples belong to the same class, where the 
    number of classes is batch_size // k.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: intra-class distances should be within this margin
        batch_size: number of embeddings 
        k: number of samples per class in embeddings
    Returns:
        loss: scalar tensor containing the anchor-center loss
    """
    loss = tf.constant(0, dtype='float32')
    for i in range(0,batch_size,k):
        anchors = embeddings[i:i+k] 
        center = tf.reduce_mean(anchors, 0)
        loss = tf.add(loss, tf.reduce_sum(tf.maximum(tf.reduce_sum(tf.square(anchors - center), axis=1) - margin, 0.)))
    return tf.reduce_mean(loss)

def __semi_hard_triplet_loss(labels, embeddings, margin = 0.2):
    return tf.contrib.losses.metric_learning.triplet_semihard_loss(labels, embeddings, margin=margin)

def __intra_enhanced_triplet_loss(labels, embeddings, lambda_1, alpha, beta, batch_size, k):
    return tf.add(__semi_hard_triplet_loss(labels, embeddings, alpha), tf.multiply(lambda_1, __anchor_center_loss(embeddings, beta, batch_size, k)))

def __large_margin_cos_loss(labels, embeddings, alpha, scale, num_cls):
    num_features = embeddings.get_shape()[1]
    
    weights = tf.get_variable("centers", [num_features, num_cls], dtype=tf.float32, 
            initializer=tf.contrib.layers.xavier_initializer(), trainable=True)

    embedds_feat_norm = tf.nn.l2_normalize(embeddings, 1, 1e-10)
    weights_feat_norm = tf.nn.l2_normalize(weights, 0, 1e-10)

    xw_norm = tf.matmul(embedds_feat_norm, weights_feat_norm)

    # value = something

    cos_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=value))
    return cos_loss 

def semi_hard_triplet_loss(margin):
    @functools.wraps(__semi_hard_triplet_loss)
    def loss(labels, embeddings):
        return __semi_hard_triplet_loss(labels, embeddings, margin)
    return loss

def intra_enhanced_triplet_loss(config):
    @functools.wraps(__intra_enhanced_triplet_loss)
    def loss(labels, embeddings):
        return __intra_enhanced_triplet_loss(labels, embeddings, config["lambda_1"], config["alpha"], config["beta"], config["batch-size"], config["k_batch"])
    return loss

def large_margin_cos_loss(config):
    @functools.wraps(__large_margin_cos_loss)
    def loss(labels, embeddings):
        return __large_margin_cos_loss(labels, embeddings, config["alpha"], config["scale"])
    return loss