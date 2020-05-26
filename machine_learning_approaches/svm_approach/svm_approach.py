from sklearn.svm import SVC
from knn_datagen import load_data
from sklearn.metrics import classification_report, confusion_matrix

train_features, train_labels = load_data('../../datasets/knn_train_data_vgg19_m.npy')

print(train_features.shape)

test_features, test_labels = load_data('../../datasets/knn_test_data_vgg19_m.npy')

svclassifier = SVC(kernel='poly', degree=2)
svclassifier.fit(train_features, train_labels)

preds = svclassifier.predict(test_features)

print(confusion_matrix(test_labels, preds))
print(classification_report(test_labels, preds))

