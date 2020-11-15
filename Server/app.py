from flask import Flask, render_template, request, redirect, url_for  # These are all we need for our purposes
import tensorflow as tf
from flask_cors import CORS
from ML.utils import predict_covid
import cv2
import numpy as np
from Server.urls import urls

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("home.html", **urls)


@app.route("/x-ray-test")
def form():
    return render_template("form.html", **urls)


@app.route("/test-result", methods=['POST'])
def predict():
    try:
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
        args = {'diagnosis': category, 'accuracy': str(accuracy) + '%'}
        for i in urls.items():
            args[i[0]] = i[1]
        return render_template("result.html", **args)

    except Exception as e:
        return redirect(url_for('error_500'))


@app.route("/error500")
def error_500():
    return render_template("error500.html", code=500)


if __name__ == "__main__":
    app.run(debug=True)
