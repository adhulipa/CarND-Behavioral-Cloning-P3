import csv
import cv2
import random
import numpy as np

import sklearn.utils


def _with_a_chance_of(prob):
    return random.random() < prob

def _is_steering_straight(angle):
    return angle <= 0.85

def _flip_l2r_randomly(image, measurement):
    if (_with_a_chance_of(0.5)):
        image = np.fliplr(image)
        measurement = -measurement
    return (image, measurement)

def init(driving_log_file):
    '''
    data_directory = '/home/carnd/projects/P3/DataFromMac/p3-training-data/track1-normal-driving/'
    driving_log_file = '/home/carnd/projects/P3/DataFromMac/p3-training-data/track1-normal-driving/driving_log.csv'
    driving_img_folder = '/home/carnd/projects/P3/DataFromMac/p3-training-data/track1-normal-driving/IMG/'
    init('/home/carnd/projects/P3/DataFromMac/p3-training-data/track1-normal-driving/driving_log.csv')
    '''
    # driving_log_file = data_directory + 'driving_log.csv'
    # driving_img_folder = data_directory + 'IMG/'

    # Read data
    lines = []
    with open(driving_log_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

def load_batch(samples, driving_img_folder, batch_size=32):
    '''
    load_batch(lines, '/home/carnd/projects/P3/DataFromMac/p3-training-data/track1-normal-driving/IMG/')
    '''
    while True:
        num_samples = len(samples)
        for offset in range(0, num_samples, batch_size):
            batch_lines = samples[offset:offset+batch_size]

        (images, measurements) = _read_images_measurements_batch(batch_lines, driving_img_folder)

        X_train = np.array(images)
        y_train = np.array(measurements)
        yield sklearn.utils.shuffle(X_train, y_train)

def _read_images_measurements_batch(batch_lines, driving_img_folder):
    images = []
    measurements = []
    for line in batch_lines:
        measurement = float(line[3])
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = driving_img_folder + filename
        image = cv2.imread(current_path)

        # Randomly flip the images
        image, measurement = _flip_l2r_randomly(image, measurement)

        images.append(image)
        measurements.append(measurement)

    return (images, measurements)