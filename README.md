# Machine-Learning
############################
Deep Learning Algorithms and Models of this repository implemented in Tensorflow and PyTorch
############################

The Proposed Optimizer and SoftMax Loss Functions Codes have bee written in Tensorflow. 

###
To call SoftMax, replace the last layer with corresponding commend:

x_1 = Dense(number_of_classes,kernel_initializer="he_normal")(input_previous layer)
out =Activation(my_forward,dynamic=True, name='ASoftMax')(x_1)
###
To call Optmizer, Use These lines of Codes to Call it:

import sklearn
import os
from AADAM import ArminAdam
ADAM_Optmizer = ArminAdam(lr=0.001)
###
To call Loss and Class-Weight Functions Use two below comment Lines:

loss=MyHuberLoss()
class_weights = create_class_weight(labels_dict)

######
