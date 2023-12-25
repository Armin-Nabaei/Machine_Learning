# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from imutils import paths
import numpy as np
import cv2


# -----------------------------
#   FUNCTIONS
# -----------------------------
def quantify_image(image, bins=(4, 6, 3)):
    # Compute a 3D color histogram over the image and normalize it
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    # Return the histogram
    return hist


def load_dataset(dataset_path, bins):
    # Grab the paths to all images in the dataset directory and initialize the lists of images
    img_paths = list(paths.list_images(dataset_path))
    data = []
    # Loop over the image paths
    for imagePath in img_paths:
        # Load the image and convert it to the HSV color space
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Quantify the image and update the data list
        features = quantify_image(image, bins)
        data.append(features)
    # Return our data list as a NumPy array
    return np.array(data)

