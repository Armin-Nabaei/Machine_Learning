import tensorflow as tf
from tensorflow import keras
import numpy as np

class MyHuberLoss(keras.losses.Loss):
    def __init__(self):
        super().__init__()
    
    def call(self, y_true, y_pred):
        threshold1 = 0.4
        threshold2 = 0.7
        batch_size = 15
        epsilon = 1.e-9
        alpha = epsilon
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        
        # Check if y_true is None, and if so, return 0 (or handle it appropriately)
        if y_true is None:
            return 0.0
        
        y_true = tf.convert_to_tensor(y_true, y_pred.dtype)
        t1 = y_true.numpy()
        lossce = keras.losses.CategoricalCrossentropy()(y_true, y_pred)

        if np.all(np.equal(t1, [1, 0, 0, 0, 0, 0, 0])):
            loss_1 = self.compute_loss(y_true, y_pred)
            return 0.1 * loss_1 + lossce

        if np.all(np.equal(t1, [0, 1, 0, 0, 0, 0, 0])):
            loss_1 = self.compute_loss(y_true, y_pred)
            return 0.7 * loss_1 + lossce

        # Add similar conditions for other cases...

    def compute_loss(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        y_true = tf.convert_to_tensor(y_true, y_pred.dtype)
        anchor = y_true
        negative = 1 - y_true
        positive = y_pred
        alpha = tf.math.reduce_max(tf.multiply(negative, y_pred))

        positive_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), -1)
        negative_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), -1)
        loss_1 = tf.add(tf.subtract(positive_dist, negative_dist), alpha)
        loss1 = tf.reduce_sum(tf.maximum(loss_1, 0.0))
        return loss1

