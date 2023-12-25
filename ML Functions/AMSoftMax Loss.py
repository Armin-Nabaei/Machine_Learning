import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.keras.losses import Loss
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.python.ops import gen_math_ops

#https://towardsdatascience.com/creating-custom-loss-functions-using-tensorflow-2-96c123d5ce6c

class MyHuberLoss(Loss): 
    def __init__(self, ):
        super().__init__()
    def call(self, y_true, y_pred,batch_size=96,num_classes=7):
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        m = 0.4
        s = 30

        kernel = tf.compat.v1.get_variable (name='kernel',dtype=tf.float32,shape=[num_classes,num_classes],initializer=tf.keras.initializers.GlorotNormal())
        kernel_norm = tf.nn.l2_normalize(kernel, 0, 1e-10, name='kernel_norm')
        cos_theta = tf.matmul(y_pred, kernel_norm)
        cos_theta = tf.clip_by_value(cos_theta, -1,1) # for numerical steady
        phi = cos_theta - m
        adjust_theta = s * tf.where(tf.equal(y_true,1), phi, cos_theta)
        loss1= keras.losses.CategoricalCrossentropy()
        err =  loss1(y_true,adjust_theta)
        return err




AMSoftMaxloss=MyHuberLoss()
