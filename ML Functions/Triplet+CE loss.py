import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.keras.losses import Loss
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.python.ops import gen_math_ops


class MyHuberLoss(Loss): #inherit parent class
  
    #initialize instance attributes
    def __init__(self, ):
        super().__init__()

    #compute loss
    def call(self, y_true, y_pred):
        v=[]
        threshold1 = 0.4
        threshold2=0.7
        #y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        #alpha=16.0  
        batch_size = 15
        epsilon = 1.e-9
        metric = []
        alpha = epsilon
        #alpha = tf.math.reduce_max ((1-y_true)*y_pred,keepdims=False)
        negative=(1-y_true)
        positive=y_pred
        anchor = y_true
        class_zero=[1,0,0,0,0,0,0]
        class_one=[0,1,0,0,0,0,0]
        class_two=[0,0,1,0,0,0,0]    
        class_three=[0,0,0,1,0,0,0] 
        class_four=[0,0,0,0,1,0,0]
        class_five=[0,0,0,0,0,1,0]
        class_six=[0,0,0,0,0,0,1]
        loss=1.0
          
           
        one = tf.constant([1,1,1,1,1,1,1], dtype=tf.float32)
        
       
       
        alpha = tf.math.reduce_max(tf.multiply((1-y_true),y_pred))
        t1=y_true.numpy()


        if (tf.reduce_all(tf.equal(t1, class_zero) )).numpy() == False:
	          positive_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), -1)
	          negative_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)), -1)
	          loss_1 = tf.add(tf.subtract(positive_dist, negative_dist), alpha)
	          loss = tf.reduce_sum(tf.maximum(loss_1,-0.0))
        else:           
            loss= keras.losses.CategoricalCrossentropy()
            loss = loss1(y_true, y_pred)

              
        return loss
