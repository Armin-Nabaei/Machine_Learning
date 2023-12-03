from pathlib import Path
import cv2
import dlib
import sys
import numpy as np
import os
import argparse
from PIL import Image
from contextlib import contextmanager
from wide_resnet import WideResNet
from tensorflow.keras.utils import get_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from os import listdir
from os.path import isfile, join
tf.compat.v1.disable_eager_execution()

epsilon = 0.40


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224,224))
    #color_mode="grayscale"
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def verifyFace(img1, img2):
    img1_representation = faceverification_descriptor.predict(preprocess_image('./training_faces/%s' % (img1)))[0,:]
    img2_representation = faceverification_descriptor.predict(preprocess_image('./training_faces/%s' % (img2)))[0,:]

    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)

    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(image.load_img('./training_faces/%s' % (img1)))
    plt.xticks([]); plt.yticks([])
    f.add_subplot(1,2, 2)
    plt.imshow(image.load_img('./training_faces/%s' % (img2)))
    plt.xticks([]); plt.yticks([])
    plt.show(block=True)

    print("Cosine similarity: ",cosine_similarity)

    if(cosine_similarity < epsilon):
        print("They are same person")
    else:
        print("They are not same person!")

img = cv2.imread('./training_faces/2.jpg', cv2.IMREAD_UNCHANGED)
dim = (48, 48)
print(img.shape)
resize=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
resize = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
img1 =cv2.imwrite('./training_faces/2.jpg',resize)

img = Image.open('./training_faces/1.jpg') 
img2 = img.convert('L')  
print(img2.size)

verifyFace("1.jpg", "2.jpg")

##
