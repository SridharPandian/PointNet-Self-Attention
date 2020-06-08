import os
import numpy as np
import h5py

from keras.utils import np_utils

def loadh5(h5_filename):
    loaded_file = h5py.File(h5_filename, 'r+')
    data = loaded_file['data'][:]
    label = loaded_file['label'][:]
    return (data, label)

def get_train_test_data(num_points, num_classes):
    path = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.abspath(os.path.join(path, os.pardir))

    # Getting training data
    train_data_path = os.path.join(base_dir, "data", "train")
    training_data_filenames = [filename for filename in os.listdir(train_data_path)]

    print('Training data directory:', train_data_path)
    print('Files present in training data directory:', training_data_filenames)

    train_points = None
    train_labels = None

    for data in training_data_filenames:
        cur_points, cur_labels = loadh5(os.path.join(train_data_path, data))
        cur_points = cur_points.reshape(1, -1, 3)
        cur_labels = cur_labels.reshape(1, -1)

        if train_labels is None or train_points is None:
            train_labels = cur_labels
            train_points = cur_points
        else:
            train_labels = np.hstack((train_labels, cur_labels))
            train_points = np.hstack((train_points, cur_points))

    train_points = train_points.reshape(-1, num_points, 3)
    train_labels = train_labels.reshape(-1, 1)

    # Getting test data
    test_data_path = os.path.join(base_dir, "data", "test")
    testing_data_filenames = [filename for filename in os.listdir(test_data_path)]

    print('Testing data directory:', test_data_path)
    print('Files present in testing data directory:', testing_data_filenames)

    test_points = None
    test_labels = None

    for data in testing_data_filenames:
        cur_points, cur_labels = loadh5(os.path.join(test_data_path, data))
        cur_points = cur_points.reshape(1, -1, 3)
        cur_labels = cur_labels.reshape(1, -1)

        if test_labels is None or test_points is None:
            test_labels = cur_labels
            test_points = cur_points
        else:
            test_labels = np.hstack((test_labels, cur_labels))
            test_points = np.hstack((test_points, cur_points))

    test_points = test_points.reshape(-1, num_points, 3)
    test_labels = test_labels.reshape(-1, 1)

    Y_train = np_utils.to_categorical(train_labels, num_classes)
    Y_test = np_utils.to_categorical(test_labels, num_classes)

    return train_points, Y_train, test_points, Y_test