# Importing data imputation and visualization libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing the Deep Learning Libraries for CNN from keras

# Using tensorflow backend
from keras.models import Sequential
from keras.layers import (
    Convolution2D, 
    MaxPooling2D,
    Flatten, 
    Dense
)   

# We need to load the dataset, images

# Initializing the CNN
classifier = Sequential()
"""
CNN's are used to analyze visual imagery,
classify images and train them
CNN - Convolutional Neural Network
"""

"""
We need to clarify about the input image size and decide on the activation function,
For this, we have 32 feature detectors with 3x3 dimensions
And should we use the relu function?? need to discuss 
"""

classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

"""
Pooling - Taken the definition from Wikipedia
Convolutional networks may include local or global pooling layers
which combine the outputs of neuron clusters at one layer into a single neuron in the next layer.
For example, max pooling uses the maximum value from each of a cluster of neurons at the 
prior layer. Another example is average pooling, which uses the average value from each of a 
cluster of neurons at the prior layer. 

Do we require pooling though, given the image is already in black and white??
"""



classifier.add(MaxPooling2D(pool_size = (2, 2)))

