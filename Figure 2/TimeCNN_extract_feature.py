from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from aeon.classification.deep_learning import TimeCNNClassifier
import tensorflow.keras as keras
import tensorflow as tf
import time


# -------------------------
# Case 1
# -------------------------

def generate_AR1_class1(n, v):
    ts = np.zeros(n)
    x_ini = np.random.normal(0, 1/v)
    for i in range(n):
        t = (i+1) / n
        if i == 0:
            ts[i] = 0.4 * np.cos(2*np.pi*t) * x_ini + np.random.normal(0, 1/v)
        else:
            ts[i] = 0.4 * np.cos(2*np.pi*t) * ts[i-1] + np.random.normal(0, 1/v)
    return ts

def generate_AR1_class2(n, v):
    ts = np.zeros(n)
    x_ini = np.random.normal(0, 1/v)
    for i in range(n):
        t = (i+1) / n
        if i == 0:
            ts[i] = 0.2 * np.cos(2*np.pi*t) * x_ini + np.random.normal(0, 1/v)
        else:
            ts[i] = 0.2 * np.cos(2*np.pi*t) * ts[i-1] + np.random.normal(0, 1/v)
    return ts


# Parameters
n_iterations_class1 = 100  # Training iterations per class
n_iterations_class2 = 100  # Training iterations per class
n_elements = 1024   # Length of each time series
n_iterations_t = 25  # Testing iterations per class
repeats = 1    # Number of times to repeat the process


# Placeholder to store all generated datasets
simunb_eps1 = []

# Repeat the process 500 times
for i in range(repeats):
    ts_class1_train = [generate_AR1_class1(n_elements, 1) for _ in range(n_iterations_class1)] 
    ts_class2_train = [generate_AR1_class2(n_elements, 1) for _ in range(n_iterations_class2)] 
    simu_eps1_train = np.array(ts_class1_train + ts_class2_train).reshape(n_iterations_class1+n_iterations_class2, 1, n_elements)

    ts_class1_test = [generate_AR1_class1(n_elements, 1) for _ in range(n_iterations_t)] 
    ts_class2_test = [generate_AR1_class2(n_elements, 1) for _ in range(n_iterations_t)] 
    simu_eps1_test = np.array(ts_class1_test + ts_class2_test).reshape(50, 1, n_elements)

    # Store the generated data for this iteration
    simunb_eps1.append({
        "train": simu_eps1_train,
        "test": simu_eps1_test
    })


arr = []
# Add for class "1"s
for _ in range(n_iterations_class1):
    arr.append('1')
# Add for class "0"s
for _ in range(n_iterations_class2):
    arr.append('0')
Y_label = np.array(arr, dtype='<U1')

arr = []
# Add for class "1"s
for _ in range(n_iterations_t):
    arr.append('1')
# Add for class "0"s
for _ in range(n_iterations_t):
    arr.append('0')
Y_test_label = np.array(arr, dtype='<U1')
Y_label = Y_label.astype('float32')
Y_test_label = Y_test_label.astype('float32')

class Classifier_CNN:

    def __init__(self, input_shape, nb_classes, verbose=False, build=True):
        # Removed output_directory
        self.verbose = verbose

        if build:
            self.model = self.build_model(input_shape, nb_classes)
            if verbose:
                self.model.summary()

    def build_model(self, input_shape, nb_classes):
        padding = 'valid'
        input_layer = keras.layers.Input(input_shape)

        if input_shape[0] < 60:  # for italypowerondemand dataset
            padding = 'same'

        # Adjust pooling sizes depending on the sequence length
        pool_size = 3 if input_shape[0] > 3 else input_shape[0]  # Make sure pooling size is smaller than input length

        conv1 = keras.layers.Conv1D(filters=6, kernel_size=7, padding=padding, activation='sigmoid')(input_layer)
        conv1 = keras.layers.AveragePooling1D(pool_size=pool_size)(conv1)

        conv2 = keras.layers.Conv1D(filters=12, kernel_size=7, padding=padding, activation='sigmoid')(conv1)
        conv2 = keras.layers.AveragePooling1D(pool_size=pool_size)(conv2)

        flatten_layer = keras.layers.Flatten()(conv2)

        # Use 1 output unit for binary classification
        output_layer = keras.layers.Dense(units=1, activation='sigmoid')(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        # Removed saving callback and file path
        self.callbacks = []

        return model

    def fit(self, x_train, y_train):
        if not tf.test.is_gpu_available:
            print('error')
            exit()

        mini_batch_size = 16
        nb_epochs = 20

        self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                       verbose=self.verbose, callbacks=self.callbacks)

    def predict(self, x_test, return_raw=True):
        # No need to load the model from a file, using the model in memory
        y_pred = self.model.predict(x_test).flatten()  # shape: (n_samples,)

        return y_pred if return_raw else (y_pred > 0.5).astype(int)


# get feature 

clf = Classifier_CNN(input_shape=(1, 1024), nb_classes=1, verbose=True)

# Train the model (only takes x_train and y_train now)
clf.fit(simu_eps1_train, Y_label)

# The feature used in boxplot
print(clf.predict(simu_eps1_test, return_raw = True))