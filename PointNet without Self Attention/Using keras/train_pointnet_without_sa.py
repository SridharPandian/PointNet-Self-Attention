# Basic Imports
import numpy as np
import tensorflow as tf

# Keras Imports
from keras.models import Model

from keras.layers import Input, Dense, Flatten, Dropout, Reshape
from keras.layers import Convolution1D, BatchNormalization, MaxPooling1D, Lambda

from keras.optimizers import Adam

# Custom imports
from functions.data_loader import get_train_test_data
from functions.other_functions import matmul, rotate_point_cloud, jitter_point_cloud

# Number of classes in the dataset
num_classes = 40

# Number of points in one point cloud 
num_points = 2048

# Input vector to be fed into the network
input_vector = Input(shape=(num_points, 3))

# Generation of 3x3 input tranformation matrix which will be used on the input vector to get a transformed input vector.
input_transform_net = Convolution1D(64, 1, activation='relu')(input_vector)
input_transform_net = BatchNormalization()(input_transform_net)

input_transform_net = Convolution1D(128, 1, activation='relu')(input_transform_net)
input_transform_net = BatchNormalization()(input_transform_net)

input_transform_net = Convolution1D(1024, 1, activation='relu')(input_transform_net)
input_transform_net = BatchNormalization()(input_transform_net)

input_transform_net = MaxPooling1D(pool_size=num_points)(input_transform_net)

input_transform_net = Dense(512, activation='relu')(input_transform_net)
input_transform_net = BatchNormalization()(input_transform_net)

input_transform_net = Dense(256, activation='relu')(input_transform_net)
input_transform_net = BatchNormalization()(input_transform_net)

input_transform_net = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(input_transform_net)

input_transform_vector = Reshape((3, 3))(input_transform_net)

# Input transformation
net = Lambda(matmul, arguments={'b': input_transform_vector})(input_vector) # Input transformation is done here

# Forward network after input transformation
net = Convolution1D(64, 1, activation='relu')(net)
net = BatchNormalization()(net)

net = Convolution1D(64, 1, activation='relu')(net)
net = BatchNormalization()(net)

# Generation of 64x64 feature transformation matrix which will be used on the feature vector to get a transformed feature vector.
feature_transform_net = Convolution1D(64, 1, activation='relu')(net)
feature_transform_net = BatchNormalization()(feature_transform_net)

feature_transform_net = Convolution1D(128, 1, activation='relu')(feature_transform_net)
feature_transform_net = BatchNormalization()(feature_transform_net)

feature_transform_net = Convolution1D(1024, 1, activation='relu')(feature_transform_net)
feature_transform_net = BatchNormalization()(feature_transform_net)

feature_transform_net = MaxPooling1D(pool_size=num_points)(feature_transform_net)

feature_transform_net = Dense(512, activation='relu')(feature_transform_net)
feature_transform_net = BatchNormalization()(feature_transform_net)

feature_transform_net = Dense(256, activation='relu')(feature_transform_net)
feature_transform_net = BatchNormalization()(feature_transform_net)

feature_transform_net = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(feature_transform_net)

feature_transform_vector = Reshape((64, 64))(feature_transform_net)

# Feature Tranformation
net = Lambda(matmul, arguments={'b': feature_transform_vector})(net)

# Forward network after feature transformation
net = Convolution1D(64, 1, activation='relu')(net)
net = BatchNormalization()(net)

net = Convolution1D(128, 1, activation='relu')(net)
net = BatchNormalization()(net)

net = Convolution1D(1024, 1, activation='relu')(net)
feature_extracted_net = BatchNormalization()(net)

# Obtaining the global features
global_feature_vector = MaxPooling1D(pool_size=num_points)(feature_extracted_net)

# Classifier network
net = Dense(512, activation='relu')(global_feature_vector)
net = BatchNormalization()(net)
net = Dropout(rate=0.7)(net)

net = Dense(256, activation='relu')(net)
net = BatchNormalization()(net)
net = Dropout(rate=0.7)(net)

net = Dense(num_classes, activation='softmax')(net)

# Prediction scores for num_classes
prediction = Flatten()(net)

model = Model(inputs = input_vector, outputs = prediction)
print(model.summary())

# Loading the data from the dataset
train_input_vector, Y_train, test_input_vector, Y_test = get_train_test_data(2048, 40)
print(train_input_vector.shape, Y_train.shape)
# print(train_input_vector[0])

# Compiling classification model
adam_optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

for custom_epoch in range(1,251):
    # Rotating the training vectors by an uniform angle
    rotated_train_vectors = rotate_point_cloud(train_input_vector)

    # Using jitter on the train point clouds
    final_train_vector = jitter_point_cloud(rotated_train_vectors)

    model.fit(final_train_vector, Y_train, batch_size=32, epochs=1, shuffle=True, verbose=1)
    print('Current epoch is: ', custom_epoch)

    # Evaluating the model with the test set after every 5 epochs
    if custom_epoch % 5 == 0 and custom_epoch != 250:
        score = model.evaluate(test_input_vector, Y_test, verbose=1)
        print('Test loss: ', score[0])
        print('Test accuracy: ', score[1])

# Final score for the model
score = model.evaluate(test_input_vector, Y_test, verbose=1)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])