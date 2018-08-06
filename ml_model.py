# Importing data imputation and visualization libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio as imio
import glob
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
imlist = []
for path in sorted(glob.glob("imagesrc/001/bg-01/*")):
    imlist_cur = []
    for im_path in sorted(glob.glob(path + "/*.png")):
        image = imio.imread(im_path)
        image = image/255
        imlist_cur.append(image)
    imlist.append(imlist_cur)


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

"""
Flattening the image
you need to convert the output of the convolutional part of the CNN into a 1D feature vector,
to be used by the ANN part of it. This operation is called flattening. It gets the output
of the convolutional layers, flattens all its structure to create a single long feature
vector to be used by the dense layer for the final classification.
"""

classifier.add(Flatten())

"""
Full Connection :-
Fully connected layers connect every neuron in one layer to every neuron in another layer.
It is in principle the same as the traditional multi-layer perceptron neural network.
"""

# First layer uses the relu activation function
classifier.add(Dense(output_dim = 128, activation = 'relu'))
# If more than 2, we use the softmax activation function,
# and the output neuron is using the softmax activation function
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

"""
We need to start working from here, after we decide all the activation functions,
and we decide how many layers and which optimization algotithm we have to use,
and whether we need to flatten the images or not
"""

# Importing the liraries so as for fitting CNN to the Images
from keras.preprocessing.image import ImageDataGenerator


