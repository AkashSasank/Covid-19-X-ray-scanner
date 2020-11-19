from ML.utils import load_dataset, predict_xray
import numpy as np
from decouple import config
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

x_train, x_test, y_train, y_test, num_classes, categories = load_dataset(config('DATASET_PATH_XRAY_NONXRAY'),
                                                                         200, 200, 0.2, one_hot_encode=False)

print(x_train.shape)
output = predict_xray(x_train, model_path='./image_classifier/best_model')
print(output)
predicted_classes = output[0]
print(predicted_classes)
print(y_train)
confusion_matrix_ = confusion_matrix(y_train, predicted_classes)
classes = ['No-X', 'X']
plt.figure('Confusion matrix')
sn.heatmap(confusion_matrix_ / np.sum(confusion_matrix_), annot=True, fmt='.2%', xticklabels=classes,
           yticklabels=classes)
plt.xlabel('True classes')
plt.ylabel('Predicted classes')
plt.show()
