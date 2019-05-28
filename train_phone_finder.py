
# Importing data manipulation modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

# Importing NN modules
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Reshape, Flatten, ReLU, Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator 
import keras.backend as K

# Directory where the images (including augmented) and labels exist 
IMG_DIR = "./aug_pics"

# Parse labels.txt to find image, convert it into array and assign
# values to the training_data variable
data = pd.read_csv("./aug_pics/aug_labels.txt", sep=" ", header=None)
data_rows = data.shape[0]
data_cols = data.shape[1]
training_data = []
for x in range(data_rows):
    ig_array = cv2.imread(os.path.join(IMG_DIR, data[0][x]))
    training_data.append([ig_array, data[1][x], data[2][x]])


# Split the training data into X input and Y output
X = []
Y = []
for features, x_cord, y_cord in training_data:
    X.append(features)
    Y.append([x_cord, y_cord]) 
X = np.array(X).reshape(-1, 326, 490, 3)
Y = np.array(Y).reshape(-1, 2)


# *******************************************************
# Building convolution network
# *******************************************************

inputs = Input(shape= X.shape[1:])

# First convolution block
l1 = Conv2D(64, (3,3), padding='same', strides=1)(inputs)
l1 = ReLU()(l1)

l2 = Conv2D(64, (3,3), padding='same', strides=1)(l1)
l2 = ReLU()(l2)

l3 = Conv2D(64, (3,3), padding='same', strides=1)(l2)
l3 = ReLU()(l3)

p1 = MaxPooling2D(pool_size=(2,2))(l3)
channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
bn1 = BatchNormalization(axis=channel_axis)(p1)
dp1 = Dropout(rate=0.2)(bn1)

# Second convolution block
l4 = Conv2D(128, (3,3), padding='same', strides=1)(dp1)
l4 = ReLU()(l4)

l5 = Conv2D(128, (3,3), padding='same', strides=1)(l4)
l5 = ReLU()(l5)

l6 = Conv2D(128, (3,3), padding='same', strides=1)(l5)
l6 = ReLU()(l6)

p2 = MaxPooling2D(pool_size=(2,2))(l6)
bn2 = BatchNormalization(axis=channel_axis)(p2)
dp2 = Dropout(rate=0.2)(bn2)

f1 = Flatten()(dp2)

# Fully connected layers
d1 = Dense(30)(f1)
bn3 = BatchNormalization(axis=channel_axis)(d1)
dp3 = Dropout(rate=0.2)(bn3)

d2 = Dense(30)(dp3)
bn4 = BatchNormalization(axis=channel_axis)(d2)
dp4 = Dropout(rate=0.2)(bn4)

d3 = Dense(2)(dp4)

model = Model(inputs=inputs, outputs=d3)

# *******************************************************
# End of neural network
# *******************************************************

model.compile(optimizer = "adam",
              loss = "mse",
              metrics = ["accuracy"])

#model.summary()
model.fit(X, Y, batch_size=5, epochs=100, validation_split=0.1, shuffle=True)

# Save the model after training
model.save("phone_detect.h5")
