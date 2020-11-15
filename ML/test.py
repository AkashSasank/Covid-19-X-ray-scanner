from ML.utils import load_dataset, predict_covid
import numpy as np
from decouple import config
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

x_train, x_test, y_train, y_test, num_classes, categories = load_dataset(config('DATASET_PATH_XRAYS'),
                                                                         200, 200, 0.2)

print(x_train.shape)
output = predict_covid(x_train)
predicted_classes = output[0]
true_classes = np.argmax(y_train, axis=1)
print(predicted_classes)
print(true_classes)
confusion_matrix_ = confusion_matrix(true_classes, predicted_classes)
classes = ['COVID-19', 'Normal', 'Viral Pneumonia']
plt.figure('Confusion matrix')
sn.heatmap(confusion_matrix_ / np.sum(confusion_matrix_), annot=True, fmt='.2%', xticklabels=classes,
           yticklabels=classes)
plt.xlabel('True classes')
plt.ylabel('Predicted classes')
plt.show()
