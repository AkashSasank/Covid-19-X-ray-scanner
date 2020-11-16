import os

from flask import Flask, render_template, request, redirect, url_for  # These are all we need for our purposes
from flask_cors import CORS

import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

from ML.utils import predict_covid
from urls import urls
from log import Logger

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(file_path)
            img = img_to_array(load_img(file_path, target_size=(200, 200))) / 255.0
            os.remove(file_path)
            img = tf.expand_dims(img, axis=0)
            # TODO : Check whether input image is X-ray or Not
            output = predict_covid(img, covid_model_path='./ML/best_covid_classifier')
            accuracy = output[2][0] * 100
            category = output[1][0]
            args = {'diagnosis': category, 'accuracy': str(accuracy) + '%'}
            for i in urls.items():
                args[i[0]] = i[1]
            return render_template("result.html", **args)
        else:
            return redirect(url_for('error_500'))

    except Exception as e:
        Logger.get_logger().exception(e)
        return redirect(url_for('error_500'))


@app.route("/error500")
def error_500():
    return render_template("error500.html", code=500)


if __name__ == "__main__":
    app.run(debug=True)
