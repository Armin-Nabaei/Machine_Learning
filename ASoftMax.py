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
from tensorflow.keras.utils import get_custom_objects
from keras.activations import *
from tensorflow.python.ops import math_ops
from tensorflow.keras.layers import Activation
import tensorflow as tf

def forward(input,m=None,axis=-1):

        numerator = tf.math.log(m)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        expm = numerator / denominator
        input = math_ops.exp(input)
        #calculate softmargin softmax
        expsum = math_ops.reduce_sum(input, axis=axis,keepdims=True) 
        input = (input*expm)/(expsum - input + ((input*expm)+0.001))
        #normalize
        normsum = math_ops.reduce_sum(input, axis=axis,keepdims=True) 
        input = tf.linalg.normalize(input/normsum)
        
        return input

  
def my_forward(input):
    
    m = tf.math.reduce_max(input)
    if m!=0:
       m = tf.math.reduce_sum (m)+0.2
       if m< -2.0 :
          m == -2.0
       if m> +2.0 :
          m == +2.0
    else:
      m=2.0
    m=np.float32(m)
    metric_value = tf.py_function(forward, [input,m], tf.float32)
    return metric_value

get_custom_objects().update({'softmax': Activation(softmax)})
