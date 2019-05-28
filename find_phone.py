# By, Nithesh Singh Sanjay | meontechno@gmail.com

# Importing data manipulation modules
import numpy as np
import os
import sys
import cv2

# Importing NN modules
from keras.models import Model, load_model

if len(sys.argv) == 2 :
    # Reading image path argument from command line 
    img_path = sys.argv[1]

    # Loading keras model
    model = load_model("saved_model/phone_detect.h5")

    # Read image into rgb numpy array
    test_img_array = cv2.imread(img_path)
    test_img_array = test_img_array.reshape(-1, 326, 490, 3)

    # Forward pass
    prediction = model.predict([test_img_array])

    print("\n\n" + str(prediction[0][0]) + " " + str(prediction[0][1]) + "\n\n")

else:
    print("\n Please enter valid image path: ./<folder>/<img.jpg> \n")