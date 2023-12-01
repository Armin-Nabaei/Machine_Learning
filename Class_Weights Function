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
# Number of labels for Each Class Setup for FER-2013
import numpy as np
import math

labels_dict = {0: 281,1: 705,2: 717,3:4772,4:2524,5:1982,6:1290} #number of each class
labels_dict2 = { 0:1.0 , 1:1.0 , 2:1.0, 3:1.0 , 4:1.0 , 5:1.0 , 6:1.0} #We can change values by choosing a value [0,1] assign the class worthinessfor computation

def create_class_weight(labels_dict,alpha=2):  # We can change alpha. The default valus of hyperparameter alpha is taken 2
    total = np.sum (list(labels_dict.values()))
    keys = labels_dict.keys()
    keys = labels_dict2.keys()
    class_weight = dict()

    numberb_training_samples=28717 # FER-2013 training samples 

    for key in keys:

        score1 =tf.math.abs( tf.math.reduce_logsumexp((alpha*(nb_train_samples/float(labels_dict[key]*labels_dict2[key])))))
        score =(tf.math.round(((score1))/int(res)).numpy())+1
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight

max_value = max(labels_dict.values())  
min_value = min(labels_dict.values())
mean=tf.math.squared_difference(max_value,min_value).numpy()+1
a = tf.constant([ mean], dtype = tf.float32)
res = tf.math.round(tf.math.sqrt(a)).numpy()
res=tf.math.log(res).numpy()
print(res)

create_class_weight(labels_dict)

