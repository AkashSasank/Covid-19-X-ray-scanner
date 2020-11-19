import os

from flask import Flask, render_template, request, redirect, url_for, \
    make_response  # These are all we need for our purposes
from flask_cors import CORS

import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

from ML.utils import predict_covid, predict_xray
from urls import urls
from log import Logger

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

urls['error_text'] = ''


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("home.html", **urls)


@app.route("/x-ray-test")
def form():
    trial = request.args.get('again')
    if trial == '1':
        urls['error_text'] = 'Try again with a valid image.'
    else:
        urls['error_text'] = ''
    response = make_response(render_template("form.html", **urls))
    response.headers.add('Cache-Control', 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0')
    return response


@app.route("/test-result", methods=['POST'])
def predict():
    try:
        image_file = request.files.get('x-ray', None)
        if image_file and allowed_file(image_file.filename):
            cwd = os.getcwd()
            if not os.path.exists(os.path.join(cwd, app.config['UPLOAD_FOLDER'])):
                os.mkdir(os.path.join(cwd, app.config['UPLOAD_FOLDER']))
            filename = secure_filename(image_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(file_path)
            img = img_to_array(load_img(file_path, target_size=(200, 200))) / 255.0
            os.remove(file_path)
            img = tf.expand_dims(img, axis=0)
            img_type = predict_xray(img, model_path='./ML/best_xray_identifier')
            # Check classification accuracy for xray_identifier
            if img_type[2][0] * 100 > 95 and img_type[0][0] == 1:
                output = predict_covid(img, covid_model_path='./ML/best_covid_classifier')
                accuracy = output[2][0] * 100
                category = output[1][0]
                args = {'diagnosis': category, 'accuracy': str(accuracy) + '%'}
                for i in urls.items():
                    args[i[0]] = i[1]
                response = make_response(render_template("result.html", **args))
                response.headers.add('Cache-Control', 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0')
                return response
            else:
                return redirect(url_for('form', **{'again': '1'}))

        else:
            return redirect(url_for('error_500'))

    except Exception as e:
        print(e)
        Logger.get_logger().exception(e)
        return redirect(url_for('error_500'))


@app.route("/error500")
def error_500():
    return render_template("error500.html", code=500)


if __name__ == "__main__":
    app.run(debug=True)