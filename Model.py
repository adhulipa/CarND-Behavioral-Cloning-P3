import csv
import cv2
import numpy as np

import pre_process

data_directory = '/home/carnd/projects/P3/DataFromMac/p3-training-data/track1-normal-driving/'
X_train, y_train = pre_process.read_data(data_directory)

# Design neural network model
from keras.models import Sequential
from keras.layers import *

def resize_function(inp):
    new_height = 66
    new_width = 200
    from keras.backend import tf as ktf
    return ktf.image.resize_images(inp, (new_height, new_width))

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3))) # Normalization layer
model.add(Cropping2D(cropping=((65, 15), (1, 1)), input_shape=(160, 320, 3)))
model.add(Lambda(resize_function)) # re-size inside the model

model.add(Convolution2D(24,5,5, input_shape=(66,200,3)))
# model.add(Convolution2D(36, 5,5))
# model.add(Convolution2D(48,5,5))

# model.add(Convolution2D(64,3,3))
# model.add(Convolution2D(64,3,3))

model.add(Flatten())

# model.add(Dense(1164))
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

# Train model
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.save('model.h5')