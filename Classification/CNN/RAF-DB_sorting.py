# Sorting RAF-DB into 7 Folders, saved in your desired root space

import os
import tensorflow as tf
import cv2
import numpy as np
from keras.models import Model  
from keras.layers import Dense,Flatten,Input  
from keras.layers import Conv2D,MaxPooling2D,Dropout,BatchNormalization  
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img 
from tensorflow.python.keras import backend as  KTF 
config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth=True 
sess = tf.compat.v1.Session(config=config) 
KTF.set_session(sess) 
import os
import shutil 
import os
import glob
from shutil import copyfile

label_path = "/list_patition_label.txt"
img_path = "/original.zip (Unzipped Files)/original"
x_train = []
y_train = []
x_test = []
y_test = []
with open(label_path) as f:
        for i in f:
                i = i.strip()
                name,cls = i.split(" ")
                img = load_img(os.path.join(img_path,name))
                img = img_to_array(img)
                if i.split("_")[0]=="train":
                       
                        if int(cls)==1:
                              cv2.imwrite('/content/drive/MyDrive/Colab Notebooks/raf-db-color/train/1/%s' % (name),img)#
                        if int(cls)==2:
                              cv2.imwrite('/content/drive/MyDrive/Colab Notebooks/raf-db-color/train/2/%s' % (name),img)  #
                        if int(cls)==3:
                              cv2.imwrite('/content/drive/MyDrive/Colab Notebooks/raf-db-color/train/3/%s' % (name),img)  #
                        if int(cls)==4:
                              cv2.imwrite('/content/drive/MyDrive/Colab Notebooks/raf-db-color/train/4/%s' % (name),img)  #
                        if int(cls)==5:
                              cv2.imwrite('/content/drive/MyDrive/Colab Notebooks/raf-db-color/train/5/%s' % (name),img)  #
                        if int(cls)==6:
                              cv2.imwrite('/content/drive/MyDrive/Colab Notebooks/raf-db-color/train/6/%s' % (name),img)  #
                        if int(cls)==7:
                              cv2.imwrite('/content/drive/MyDrive/Colab Notebooks/raf-db-color/train/7/%s' % (name),img)  #

                if i.split("_")[0]=="test":
                        
                        if int(cls)==1:
                              cv2.imwrite('/content/drive/MyDrive/Colab Notebooks/raf-db-color/validation/1/%s' % (name),img)#
                        if int(cls)==2:
                              cv2.imwrite('/content/drive/MyDrive/Colab Notebooks/raf-db-color/validation/2/%s' % (name),img)  #
                        if int(cls)==3:
                              cv2.imwrite('/content/drive/MyDrive/Colab Notebooks/raf-db-color/validation/3/%s' % (name),img)  #
                        if int(cls)==4:
                              cv2.imwrite('/content/drive/MyDrive/Colab Notebooks/raf-db-color/validation/4/%s' % (name),img)  #
                        if int(cls)==5:
                              cv2.imwrite('/content/drive/MyDrive/Colab Notebooks/raf-db-color/validation/5/%s' % (name),img)  #
                        if int(cls)==6:
                              cv2.imwrite('/content/drive/MyDrive/Colab Notebooks/raf-db-color/validation/6/%s' % (name),img)  #
                        if int(cls)==7:
                              cv2.imwrite('/content/drive/MyDrive/Colab Notebooks/raf-db-color/validation/7/%s' % (name),img)  #


