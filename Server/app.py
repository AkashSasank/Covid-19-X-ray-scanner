from flask import Flask, render_template, request  # These are all we need for our purposes
import tensorflow as tf
from flask_cors import CORS
from ML.utils import predict_covid
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("home.html", **{"test_url": "http://127.0.0.1:5000/x-ray-test"})


@app.route("/x-ray-test")
def form():
    return render_template("form.html", **{"results_url": "http://127.0.0.1:5000/test-result"})


@app.route("/test-result", methods=['POST'])
def predict():
    image_file = request.files.get('x-ray', None)
    img = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    if len(img.shape) <= 2:
        img = cv2.merge((img, img, img))

    img = cv2.resize(img, (200, 200)) / 255.0
    img = tf.expand_dims(img, axis=0)
    # TODO : Check whether input image is X-ray or Not
    output = predict_covid(img, covid_model_path='../ML/best_covid_classifier')
    accuracy = output[2][0] * 100
    category = output[1][0]

    return render_template("result.html", **{'diagnosis': category, 'accuracy':str(accuracy)+'%'})


if __name__ == "__main__":
    app.run(debug=True)
