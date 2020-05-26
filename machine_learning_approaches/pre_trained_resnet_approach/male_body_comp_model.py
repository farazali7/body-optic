import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dropout, Dense, Flatten, Activation
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.optimizers import SGD, Adam
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import cv2

BATCH_SIZE = 5
EPOCHS = 8

X = pickle.load(open("male_data/X.pickle", "rb"))
y = pickle.load(open("male_data/y.pickle", "rb"))

# Normalize image vectors
X = X/255.0

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

resnet_base = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224,224,3))

for layer in resnet_base.layers:
    layer.trainable = False

x = resnet_base.output
x = layers.GlobalAveragePooling2D()(x)
x = Dropout(0.1)(x)
predictions = layers.Dense(11, activation='softmax')(x)

model = Model(resnet_base.input, predictions)

adam = Adam(lr=0.001)
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# tensorboard = TensorBoard(log_dir="logs\{}".format(time()))

train_datagen = ImageDataGenerator(
    rotation_range=40,
    shear_range=0.5,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=(0.5, 2)
)

# model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE), epochs=EPOCHS)

preds = model.evaluate(X_test, y_test)
print('Loss = ' + str(preds[0]))
print('Test Accuracy = ' + str(preds[1]))



