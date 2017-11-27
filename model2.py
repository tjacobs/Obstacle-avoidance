from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Lambda, Dropout
from keras.layers import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

# What the model expects
image_width = 320
image_height = 160

# Define our model
def Model():

    # Define model
    model = Sequential()

    # Crop
    model.add(Cropping2D(cropping=((20, 20), (0, 0)), input_shape=(image_height, image_width, 3)))

    # Five convolution layers
    model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2), padding="valid"))
    model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2), padding="valid"))
    model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2), padding="valid"))
    model.add(Conv2D(64, (3, 3), activation="relu", strides=(1, 1), padding="valid"))
    model.add(Conv2D(64, (3, 3), activation="relu", strides=(1, 1), padding="valid"))

    # One flatten layer
    model.add(Flatten())

    # Five fully connected layers
    model.add(Dense(400, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    return model
    
