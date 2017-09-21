import csv
import cv2
import numpy as np

CURRENT_DIR = '/home/carnd/projects/P3-CarND-Behavioral-Cloning/'

# Read data
lines = []
with open(CURRENT_DIR + 'data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = CURRENT_DIR + 'data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurements.append(float(line[3]))

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
