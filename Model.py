import csv
import cv2
import numpy as np

from pre_process import read_data

data_directory = '/home/carnd/projects/P3/DataFromMac/p3-training-data/track1-normal-driving/'
images, measurements = read_data(data_directory)

# Define inputs to model
X_train = np.array(images)
y_train = np.array(measurements)


# Design neural network model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# Train model
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.save('model.h5')
