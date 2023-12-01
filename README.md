# Machine-Learning
############################
Deep Learning Algorithms and Models of this repository implemented in Tensorflow and PyTorch
############################

The Proposed Optimizer and SoftMax Loss Functions Codes have bee written in Tensorflow. 

###
To call SoftMax, replace the last layer with corresponding commend:

x_1 = Dense(number_of_classes,kernel_initializer="he_normal")(input_previous layer)
out =Activation(my_forward,dynamic=True, name='ASoftMax')(x_1)
<img width="467" alt="Screenshot 2023-11-30 at 1 05 20 PM" src="https://github.com/arminn84/Machine-Learning/assets/150948007/c831a9c2-c351-420a-b7a2-7f679dd00be8">
<img width="475" alt="Screenshot 2023-11-30 at 1 05 38 PM" src="https://github.com/arminn84/Machine-Learning/assets/150948007/ba8af3ad-f92e-422f-b1eb-b8c906d553d8">


###
To call Optmizer, Use These lines of Codes to Call it:

import sklearn
import os
from AADAM import ArminAdam
ADAM_Optmizer = ArminAdam(lr=0.001)
###
<img width="437" alt="Screen Shot 2023-09-28 at 12 29 05 AM" src="https://github.com/arminn84/Machine-Learning/assets/150948007/d085d97c-d1b7-4309-908c-8dc1736cb86e">

################

To call Loss and Class-Weight Functions Use two below comment Lines:

loss=MyHuberLoss()
class_weights = create_class_weight(labels_dict)

######
<img width="360" alt="Screenshot 2023-11-30 at 1 00 13 PM" src="https://github.com/arminn84/Machine-Learning/assets/150948007/c3554520-362f-4473-b642-926b682ee8a1">
<img width="358" alt="Screenshot 2023-11-30 at 1 00 23 PM" src="https://github.com/arminn84/Machine-Learning/assets/150948007/3a5bd62e-0450-429f-82a8-07af765b1c4c">


