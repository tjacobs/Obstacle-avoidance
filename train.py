import tensorflow as tf
import numpy as np
from scipy.misc import imsave
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from model2 import WaypointsModel, image_width, image_height
import keras

# Start
print("Running on TensorFlow", tf.__version__, "Keras", keras.__version__)

# Set configuration
train_data_dir = 'train'
validation_data_dir = 'validation'
nb_train_samples = 1000
nb_validation_samples = 100
epochs = 10
batch_size = 30

# Compile
model = WaypointsModel()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())

# Shear, zoom, and flip training data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255)
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True)

# Just rescale color for validation data
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

classes = ['clear', 'obstacle']

# Training data
print("Training:")
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    classes=classes,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical')

# Validation data
print("Validation:")
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    classes=classes,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical')

def batch_generator(generator):
    while True:
        images, classifications = generator.next()

        waypoints = []
        for classification in classifications:
            # Clear road
            if classification[0] == 1:
                waypoints.append([0,0,0,0,0])
            # Obstacle road
            else:
                waypoints.append([0.0,0.1,0.3,0.5,0.2])

        # Yield  image, [ [1, 0], [waypoints] ]
        yield(images, [classifications, np.array(waypoints)])
#        yield(image, [category, (np.array([waypoints]*category.shape[0]))])

# i = 0
# for sample in batch_generator(train_generator):
#     print(sample)
# #    imsave('img' + str(i) + '.jpg', sample[0][0]) # First sample of the batch, first image
#     i += 1
#     if i > 2:
#         break


# Callbacks
checkpoint = ModelCheckpoint(filepath='./model_checkpoint.h5', verbose=1, save_best_only=True)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
callbacks = []#[checkpoint, lr_reducer]

# Train
model.fit_generator(
    batch_generator(train_generator),
    steps_per_epoch = nb_train_samples // batch_size,
    validation_data = batch_generator(validation_generator),
    validation_steps = nb_validation_samples // batch_size,
    epochs=epochs,
    callbacks=callbacks)

print("Saving...")

# Save
model.save('detect_model.h5')

print("Saved.")
