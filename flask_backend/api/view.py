from flask import Blueprint, request
from knn_classifier import make_prediction

main = Blueprint('main', __name__)

@main.route('/send_image', methods=['POST'])
def send_image():
    data = request.files['image']
    print(data)

    return 'Done', 201
