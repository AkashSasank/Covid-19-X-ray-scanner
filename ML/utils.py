import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import random
from tensorflow.keras.models import load_model
import tensorflow as tf
import autokeras as ak
from keras.activations import softmax


def load_dataset(dataset_dir: str = None,
                 image_height: int = 100,
                 image_width: int = 100,
                 test_split: float = 0.1,
                 one_hot_encode=True,
                 num_samples: int = 100, gray=False):
    X = []
    Y = []
    categories = os.listdir(dataset_dir)
    for index, category in enumerate(categories):
        folder = os.path.join(dataset_dir, category)
        files = os.listdir(folder)
        sample_count = num_samples
        if sample_count > len(files):
            sample_count = len(files)
        files = random.sample(files, sample_count)

        for file_name in files:
            try:
                image = cv2.imread(os.path.join(folder, file_name))
                if gray:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image = cv2.resize(image, (image_width, image_height))
                X.append(image)
                Y.append(int(index))
            except Exception as e:
                print(e)

    X = np.array(X, dtype=np.float) / 255
    Y = np.array(Y)
    num_class = len(categories)
    if one_hot_encode:
        Y = to_categorical(Y, num_class)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_split)
    return X_train, X_test, Y_train, Y_test, num_class, categories


def predict_covid(image, covid_model_path='./best_covid_classifier'):
    classes = ['COVID-19', 'Normal', 'Viral Pneumonia']
    loaded_model = load_model(covid_model_path, custom_objects=ak.CUSTOM_OBJECTS)
    # loaded_model.summary()
    predicted_y = loaded_model.predict(tf.expand_dims(image, -1))
    predictions = np.argmax(predicted_y, axis=1)
    categories = [classes[j] for index, j in enumerate(predictions)]
    probablities = [predicted_y[index][j] for index, j in enumerate(predictions)]
    return (predictions, categories, probablities)


def predict_xray(image, model_path='./best_xray_identifier'):
    classes = ['No-X', 'X']
    loaded_model = load_model(model_path, custom_objects=ak.CUSTOM_OBJECTS)
    # loaded_model.summary()
    predicted_y = loaded_model.predict(tf.expand_dims(image, -1)) * 100
    predictions = [1 if j[0] > .5 else 0 for index, j in enumerate(predicted_y)]
    categories = [classes[j] for index, j in enumerate(predictions)]
    probablities = [j[0] for index, j in enumerate(predicted_y)]
    return (predictions, categories, probablities)
