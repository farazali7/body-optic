import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet_v2 import ResNet50V2

# Load raw train data
train_data = np.load('../datasets/female_dataset_train.npy')

# Extract feature vectors from ConvNet

def get_vgg19():
    model = VGG19(weights='imagenet', include_top=False)
    return model

def get_vgg16():
    model = VGG16(weights='imagenet', include_top=False)
    return model

def get_resnet():
    model = ResNet50V2(weights='imagenet', include_top=False)
    return model

feature_extractor = get_vgg19()

# Create list of feature vectors and labels
train_feature_vector_list = []
train_label_list = []

count = 0
group = 0
for cls in train_data:
    if count<len(train_data)-1:

        for img_array in cls:
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.vgg19.preprocess_input(img_array)

            img_feature = feature_extractor.predict(img_array)
            img_feature_array = np.array(img_feature)

            train_feature_vector_list.append(img_feature_array.flatten())
            train_label_list.append(group)

        if count % 2 != 0:
            group = group + 1

        count = count + 1

train_feature_vector_array = np.array(train_feature_vector_list)

print(train_feature_vector_array.shape)
print(np.array(train_label_list).shape)
print(train_label_list)

train_features_and_labels = list(zip(train_feature_vector_array, train_label_list))
random.Random(4).shuffle(train_features_and_labels)

np.save("../datasets/knn_train_data_vgg19_f", train_features_and_labels)

# Load raw test data
test_data = np.load('../datasets/female_dataset_test.npy')

test_feature_vector_list = []
test_label_list = []

count = 0
group = 0
for cls in test_data:
    if count < len(test_data)-1:

        for img_array in cls:
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.vgg19.preprocess_input(img_array)

            img_feature = feature_extractor.predict(img_array)
            img_feature_array = np.array(img_feature)

            test_feature_vector_list.append(img_feature_array.flatten())
            test_label_list.append(group)

        if count % 2 != 0:
            group = group + 1

        count = count + 1

test_feature_vector_array = np.array(test_feature_vector_list)

print(test_feature_vector_array.shape)
print(np.array(test_label_list).shape)

test_features_and_labels = list(zip(test_feature_vector_array, test_label_list))
random.Random(4).shuffle(test_features_and_labels)

np.save("../datasets/knn_test_data_vgg19_f", test_features_and_labels)

def load_data(path):
    features_and_labels = np.load(path, allow_pickle=True)
    features = list(zip(*features_and_labels))[0]
    features = np.array(features)
    labels = list(zip(*features_and_labels))[1]
    labels = np.array(labels)

    return features, labels


