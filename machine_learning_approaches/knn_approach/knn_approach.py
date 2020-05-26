from sklearn.model_selection import cross_val_score
from knn_datagen import load_data
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn import metrics

train_features, train_labels = load_data('datasets/knn_train_data_vgg19_f.npy')

print(train_features.shape)

test_features, test_labels = load_data('datasets/knn_test_data_vgg19_f.npy')

# Create classifier
neighbours = list(range(1, 29))

cv_scores = []

nca = make_pipeline(StandardScaler(), NeighborhoodComponentsAnalysis(1))

nca.fit(train_features, train_labels)

for k in neighbours:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, nca.transform(train_features), train_labels, cv=8, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
mse = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbours[mse.index(min(mse))]
print("The optimal number of neighbors is {}".format(optimal_k))

# plot misclassification error vs k
plt.plot(neighbours, mse)
plt.xlabel("Number of Neighbors K")
plt.ylabel("Misclassification Error")
plt.show()

new_knn = KNeighborsClassifier(8)
new_knn.fit(nca.transform(train_features), train_labels)
print(test_features.shape)

pred_labels = new_knn.predict(nca.transform(test_features))

print(pred_labels)
print(test_labels)
print('Accuracy:', metrics.accuracy_score(test_labels, pred_labels))

# One extra base test image (should output 0)
# image = load_img(r"C:\Users\Faraz\Desktop\test10.jpg", target_size=(224, 224))
# test_arr = img_to_array(image)
# test_arr = test_arr.reshape((1,) + test_arr.shape)
# test_arr = preprocess_input(test_arr)
#
# print(test_arr.shape)
#
# model = get_vgg19()
#
# test_feature_final = model.predict(test_arr)
#
# test_feature_final = np.array(test_feature_final).flatten()
# test_feature_final = test_feature_final.reshape((1,) + test_feature_final.shape)
#
# final_label = new_knn.predict(nca.transform(test_feature_final))

# print(final_label)
