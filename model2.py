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

from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

# What the model expects
image_width = 320
image_height = 160

# Define our model
def WaypointsModel():

    # Define model input
    main_input = Input(shape=(image_height, image_width, 3), name='main_input')

    # Crop
    x = Cropping2D(cropping=((20, 20), (0, 0)))(main_input)

    # Five convolution layers
    x = Conv2D(24, (5, 5), activation="relu", strides=(2, 2), padding="valid")(x)
    x = Conv2D(36, (5, 5), activation="relu", strides=(2, 2), padding="valid")(x)
    x = Conv2D(48, (5, 5), activation="relu", strides=(2, 2), padding="valid")(x)
    x = Conv2D(64, (3, 3), activation="relu", strides=(1, 1), padding="valid")(x)
    x = Conv2D(64, (3, 3), activation="relu", strides=(1, 1), padding="valid")(x)

    # One flatten layer
    x = Flatten()(x)

    # Five fully connected layers
    x = Dense(400, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    output1 = Dense(2, activation='softmax', name='output1')(x)
    output2 = Dense(5, name='output2')(x)

    # Create
    model = Model(inputs=main_input, outputs=[output1, output2])
    return model
    
