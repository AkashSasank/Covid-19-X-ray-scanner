from tensorflow.keras.models import load_model
import tensorflow as tf
import autokeras as ak
from ML.utils import load_dataset
import numpy as np
from decouple import config

x_train, x_test, y_train, y_test, num_classes, categories = load_dataset(config('DATASET_PATH_XRAY_NONXRAY'),
                                                                         200, 200, 0.2)

loaded_model = load_model("./best_xray_identifier", custom_objects=ak.CUSTOM_OBJECTS)

predicted_y = loaded_model.predict(tf.expand_dims(x_test, -1))

predictions = np.argmax(predicted_y, axis=1)
probablities = [predicted_y[index][j] for index, j in enumerate(predictions)]
print(predicted_y.shape)
print(predictions.shape)
print(probablities)
