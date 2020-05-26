import numpy as np
import os
import cv2
import random
import pickle
import shutil

DATADIR = r"raw_dataset"
IMG_SIZE = 224

def create_training_data(gender):
    training_data = []
    percentages = []

    spec_path = os.path.join(DATADIR, gender.lower())

    if gender.lower() == 'male':
        percentages = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    elif gender.lower() == "female":
        percentages = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

    for percent in percentages:
        path = os.path.join(spec_path, str(percent))
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))

            # Convert BGR loaded image back to RGB and resize for ResNet
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

            # Scale the labels to 0-11 based on min percentage
            min = percentages[0]

            training_data.append([img_array, percent - min])

    # Create separate arrays for images (features) and body fat labels
    X = []
    y = []

    random.shuffle(training_data)

    for features, labels in training_data:
        X.append(features)
        y.append(labels)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    y = np.array(y)

    # Save training data

    save_path = gender.lower() + "_" + "data"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)

    with open(os.path.join(save_path, "X.pickle"), "wb") as pickle_out:
        pickle.dump(X, pickle_out)

    with open(os.path.join(save_path, "y.pickle"), "wb") as pickle_out:
        pickle.dump(y, pickle_out)

create_training_data("Male")