"""
Custom loss functions for machine learning models in Keras/TensorFlow.
Includes binary and categorical focal loss functions.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import dill

# Constants
GAMMA_DEFAULT = 2.0
ALPHA_DEFAULT = 0.25
EPSILON = K.epsilon()

def binary_focal_loss(gamma: float = GAMMA_DEFAULT, alpha: float = ALPHA_DEFAULT) -> tf.Tensor:
    """
    Binary form of the focal loss function.

    Arguments:
    gamma -- focusing parameter to minimize easy-to-classify examples.
    alpha -- balancing factor for class imbalance.

    Returns:
    A callable loss function.
    """
    def binary_focal_loss_fixed(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Binary focal loss function.

        Arguments:
        y_true -- tensor of true labels.
        y_pred -- tensor of predicted labels.

        Returns:
        Loss tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = K.clip(y_pred, EPSILON, 1.0 - EPSILON)
        
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_t, 1 - alpha_t)
        
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        
        loss = weight * cross_entropy
        return K.mean(K.sum(loss, axis=1))

    return binary_focal_loss_fixed

def categorical_focal_loss(alpha: np.ndarray, gamma: float = GAMMA_DEFAULT) -> tf.Tensor:
    """
    Softmax version of the focal loss function for multi-class classification problems.

    Arguments:
    alpha -- array of weights for class imbalance.
    gamma -- focusing parameter to minimize easy-to-classify examples.

    Returns:
    A callable loss function.
    """
    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Categorical focal loss function.

        Arguments:
        y_true -- tensor of true labels.
        y_pred -- tensor of predicted labels.

        Returns:
        Loss tensor.
        """
        y_pred = K.clip(y_pred, EPSILON, 1.0 - EPSILON)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed

def serialize_test():
    """
    Tests the serialization of the custom loss functions.
    """
    binary_loss_serialized = dill.loads(dill.dumps(binary_focal_loss(gamma=GAMMA_DEFAULT, alpha=ALPHA_DEFAULT)))
    categorical_loss_serialized = dill.loads(dill.dumps(categorical_focal_loss(gamma=GAMMA_DEFAULT, alpha=ALPHA_DEFAULT)))

    # Use logging instead of print for production-level code
    logging.info(binary_loss_serialized)
    logging.info(categorical_loss_serialized)

if __name__ == '__main__':
    serialize_test()
