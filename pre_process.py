import csv
import cv2
import random
import numpy as np

import sklearn.utils
from sklearn.model_selection import train_test_split


def _with_a_chance_of(prob):
    return random.random() < prob

def _is_steering_straight(angle):
    return angle <= 0.85

def _flip_l2r_randomly(image, measurement):
    if (_with_a_chance_of(0.5)):
        image = np.fliplr(image)
        measurement = -measurement
    return (image, measurement)

def read_data(data_directory, batch_size=19283719287312):
    '''
    data_directory = '/home/carnd/projects/P3/DataFromMac/p3-training-data/track1-normal-driving/'
    driving_log_file = '/home/carnd/projects/P3/DataFromMac/p3-training-data/track1-normal-driving/driving_log.csv'
    driving_img_folder = '/home/carnd/projects/P3/DataFromMac/p3-training-data/track1-normal-driving/IMG/'
    read_data(data_directory)
    '''

    driving_log_file = data_directory + 'driving_log.csv'
    driving_img_folder = data_directory + 'IMG/'

    # Read data
    lines = []
    with open(driving_log_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    train_lines, validation_lines = train_test_split(lines, test_size=0.2)

    num_samples = len(lines)
    for offset in range(0, num_samples, batch_size):
        batch_lines = lines[offset:offset+batch_size]

    (images, measurements) = _read_batch(batch_lines, driving_img_folder)

    X_train = np.array(images)
    y_train = np.array(measurements)

    return (X_train, y_train)

def _read_batch(batch_lines, driving_img_folder):
    images = []
    measurements = []
    for line in batch_lines:
        measurement = float(line[3])

        # Exclude like 70% of the data where the car is going straight
        if (_is_steering_straight(measurement) and _with_a_chance_of(0.7)):
            continue

        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = driving_img_folder + filename
        image = cv2.imread(current_path)

        # Randomly flip the images
        image, measurement = _flip_l2r_randomly(image, measurement)

        images.append(image)
        measurements.append(measurement)

    return (images, measurements)