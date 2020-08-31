#
# * Basic classification: Classify images of clothing

#%%
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# ? Fashion MNIST is replacement of classic MNIST dataset - often used as the
# ? "Hello world" of machine learning for computer vision.
fashion_mnist = keras.datasets.fashion_mnist

# ? train_images and train_labels arrays are the training set
# ? test_images and test_labels arrays are test set
# ? images are 28 x 28 Numpy arrays, with pixel values ranging from 0 to 255.
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouse', 'Pullover', 'Dress', 'Coat', 'Sandal',
    'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)

print(len(train_labels))
