from create_train_val_test_surge import load_surge_data
import numpy as np
from matplotlib import pyplot as plt

X_train, y_train, X_test, y_test, X_val, y_val = load_surge_data("../surge_prediction_data", "all")

surge_ammounts = []

for row in y_train:
    surge_ammounts += row.tolist()

plt.hist(surge_ammounts, log=True, bins=50)
plt.show()


def surge_threshold(y):
    pos = np.where(y > 75, 1, 0)
    y = pos
    return y


surge_threshold(y_val)