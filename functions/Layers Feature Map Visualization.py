import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.framework import ops
ops.reset_default_graph()
from tensorflow.keras import backend as K
tf.disable_v2_behavior()

######
plt.matshow(first_layer_activation[ :, :, 1,1], cmap='viridis')

####
for i in range(28,32):

    plt.matshow(first_layer_activation[:, :,0,i], cmap='viridis')
####

layer_names = []
for layer in model.layers[:9]:
    layer_names.append(layer.name)
images_per_row = 16

# Get CONV layers only
conv_layer_names = []
for layer_name in layer_names:
    if 'conv2d' in layer_name:
        conv_layer_names.append(layer_name)

for layer_name, layer_activation in zip(conv_layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]

    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

last_conv_layer = model.get_layer('conv2d_4')
tr_output = model.output[:, :]
np.argmax(activations[0])

tiger_output = model.output[:,:]
last_conv_layer = model.get_layer('conv2d_4')
######

# Gradients of the Tiger class wrt to the block5_conv3 filer
grads = K.gradients(tiger_output, last_conv_layer.output)[0]
# Each entry is the mean intensity of the gradient over a specific feature-map channel
pooled_grads = K.mean(grads, axis=(3))
print(grads)
print(pooled_grads)
#####

# Accesses the values we just defined given our sample image
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])




