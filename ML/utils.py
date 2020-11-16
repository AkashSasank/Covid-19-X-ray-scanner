import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import autokeras as ak


def predict_covid(image, covid_model_path='./best_covid_classifier'):
    classes = ['COVID-19', 'Normal', 'Viral Pneumonia']
    loaded_model = load_model(covid_model_path, custom_objects=ak.CUSTOM_OBJECTS)
    # loaded_model.summary()
    predicted_y = loaded_model.predict(tf.expand_dims(image, -1))
    predictions = np.argmax(predicted_y, axis=1)
    categories = [classes[j] for index, j in enumerate(predictions)]
    probablities = [predicted_y[index][j] for index, j in enumerate(predictions)]
    return (predictions, categories, probablities)
