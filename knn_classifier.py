from keras_preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
from knn_datagen import load_data, get_vgg19
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

# VALUES FOUND FROM PRIOR CV/TESTS
GENDER_BASED_K = {
    "male": 7,
    "female": 8,
}

GENDER_BASED_DATA = {
    "male": '../../datasets/knn_train_data_vgg19_m.npy',
    "female": '../../datasets/knn_train_data_vgg19_f.npy'
}

def make_prediction(image_url, gender):
    data_endpoint = GENDER_BASED_DATA["male"]
    k_value = GENDER_BASED_K["male"]

    if gender.lower() == 'male':
        data_endpoint = GENDER_BASED_DATA["male"]
        k_value = GENDER_BASED_K["male"]

    elif gender.lower() == 'female':
        data_endpoint = GENDER_BASED_DATA["female"]
        k_value = GENDER_BASED_K["female"]

    train_features, train_labels = load_data(data_endpoint)

    nca = make_pipeline(StandardScaler(), NeighborhoodComponentsAnalysis(1))

    nca.fit(train_features, train_labels)

    knn = KNeighborsClassifier(k_value)
    knn.fit(nca.transform(train_features), train_labels)

    # Loading and converting in image
    image = load_img(image_url, target_size=(224, 224))
    test_arr = img_to_array(image)
    test_arr = test_arr.reshape((1,) + test_arr.shape)
    test_arr = preprocess_input(test_arr)

    model = get_vgg19()

    test_feature_final = model.predict(test_arr)

    test_feature_final = np.array(test_feature_final).flatten()
    test_feature_final = test_feature_final.reshape((1,) + test_feature_final.shape)

    final_label = knn.predict(nca.transform(test_feature_final))

    predicted_value = final_label[0] + 10

    return predicted_value



