#Copyright [2023] [ARMIN NABAEI]

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
==============================================================================
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.keras.losses import Loss
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.python.ops import gen_math_ops

class MyLoss(Loss): 

    def __init__(self, ):
        super().__init__()
    def call(self, y_true, y_pred):
        v=[]
        threshold1 = 0.3
        threshold2=0.7
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        alpha=4.0
        batch_size = 32
        epsilon = 1.e-9
        metric = []

        if    (threshold2 >= threshold1) :
              loss1= keras.losses.CategoricalCrossentropy()
              big_error_loss_2 = loss1(y_true, y_pred)
              model_out = tf.add(y_pred, epsilon)
              weight =  tf.multiply(y_true,tf.pow(tf.subtract(1., model_out),2.0)) 
              alpha=4.0
              numerator = tf.math.log (model_out)
              denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
              model_out = numerator / denominator
              ce = tf.multiply(y_true, model_out) 
              fl = tf.math.negative(tf.multiply(alpha, (tf.multiply((weight), ce))))
              small_error_loss = tf.reduce_max(fl, axis=1)
              
              return 0.5 * big_error_loss_2 + 0.5 * small_error_loss



