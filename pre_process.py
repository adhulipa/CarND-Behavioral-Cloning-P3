import csv
import cv2

def read_data(data_directory):
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

    images = []
    measurements = []
    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = driving_img_folder + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurements.append(float(line[3]))
    return (images, measurements)