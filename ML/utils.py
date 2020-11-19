from tensorflow.keras.models import load_model
import tensorflow as tf


def predict_covid(image, covid_model_path='./best_covid_classifier'):
    classes = ['COVID-19', 'Normal', 'Viral Pneumonia']
    loaded_model = load_model(covid_model_path)
    # loaded_model.summary()
    predicted_y = loaded_model.predict(tf.expand_dims(image, -1))
    predictions = tf.argmax(predicted_y, axis=1)
    categories = [classes[j] for index, j in enumerate(predictions)]
    probablities = [predicted_y[index][j] for index, j in enumerate(predictions)]
    return (predictions, categories, probablities)


def predict_xray(image, model_path='./best_xray_identifier'):
    classes = ['No-X', 'X']
    loaded_model = load_model(model_path)
    # loaded_model.summary()
    predicted_y = loaded_model.predict(tf.expand_dims(image, -1)) * 100
    predictions = [1 if j[0] > .5 else 0 for index, j in enumerate(predicted_y)]
    categories = [classes[j] for index, j in enumerate(predictions)]
    probablities = [j[0] for index, j in enumerate(predicted_y)]
    return (predictions, categories, probablities)
