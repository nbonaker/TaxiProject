import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from create_train_val_test_surge import load_surge_data
import numpy as np
from matplotlib import pyplot as plt

model = keras.Sequential(
    [
        keras.Input(shape=(273,)),
        layers.Dense(500, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1000, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(500, activation='relu'),
        layers.Dense(266),
    ]
)
print(model.summary())

model.compile(
    optimizer='adam',  # Optimizer
    # Loss function to minimize
    loss=keras.losses.MeanSquaredError(),
    # List of metrics to monitor
    metrics=[keras.metrics.MeanSquaredError()],
)

X_train, y_train, X_test, y_test, X_val, y_val = load_surge_data("../surge_prediction_data", "all")

history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=10,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(X_val, y_val),
)

print("Evaluate on test data")
results = model.evaluate(X_test, y_test)
print("test loss, test acc:", results)

for i in [1, 15, 444, 240]:
    preds = model.predict(np.array([X_test[i]]))[0]
    targets = y_test[i]

    combined = list(zip(preds, targets))
    combined.sort(key=lambda x: x[1])
    preds, targets = list(zip(*combined))

    plt.bar(range(1, 267), targets, alpha=0.7)
    plt.bar(range(1, 267), preds, alpha=0.7)

    plt.show()