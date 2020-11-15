from ML.utils import load_dataset
from keras.preprocessing.image import ImageDataGenerator
import autokeras as ak
from decouple import config
from tensorflow.keras.callbacks import TensorBoard

x_train, x_test, y_train, y_test, num_classes, categories = load_dataset(config('DATASET_PATH_XRAY_NONXRAY')
                                                                         , 200, 200,
                                                                         0.2, num_samples=200)
# Initialize the image classifier.
clf = ak.ImageClassifier(
        overwrite=True,
        max_trials=1)
tb = TensorBoard(histogram_freq=1, embeddings_freq=1)
print(y_train.shape)
# Feed the image classifier with training data.
clf.fit(x_train, y_train, epochs=50, callbacks=[tb])

# Predict with the best model.
predicted_y = clf.predict(x_test)
print(predicted_y)

# Evaluate the best model with testing data.
print(clf.evaluate(x_test, y_test))

model = clf.export_model()

print(type(model))  # <class 'tensorflow.python.keras.engine.training.Model'>

try:
    model.save("best_xray_identifier", save_format="tf")
except:
    model.save("best_xray_identifier.h5")
