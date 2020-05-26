<p align="center">
    <img src="web_appv2\src\assets\images\logo1.png">
</p>

---

BodyOptic is a web application that allows a user to take a front-facing digital photograph of themselves and then upload it to a machine learning algorithm to determine their body composition. The app uses a pre-trained VGG19 convolutional neural network without its final output layer, thus, rendering it as just a feature extractor for the given image. This data is then fed to a K-NearestNeighbour classifier to classifying the user's body fat percentage. The data used for voting (comparison) is composed of images of males and females ranging from 10-20 and 22-32 body fat percentages respecively. These dataset images were taken from individuals who presented an accompanying DEXA scan report, which is considered the "gold standard" of measuring body composition.

>Note: At no point are images given to application stored or used in any manner outside of application's intended use. 

# Current Progress
The model currently acheives 40% classification accuracy for male images and 30% for females. Many different data processing/machine learning approaches have been tested to get to this level of accuracy for such a small dataset (5 images per percentage per gender), and can be found under the `machine_learning_approaches` directory. Using React and Javascript, a simple web application has also been set up, however, there is still work to do for complete connection with backend. 

# To-Dos
- Research, test, and implement more approaches to find better accuracy
- Find more raw data to increase dataset size
- Finish connection of React frontend web app with Flask backend + add instructional details to frontend
- Convert web app to a Progressive Web App for installation on iOS and Android devices (+ optimize for camera usage)

# Author
- Faraz Ali