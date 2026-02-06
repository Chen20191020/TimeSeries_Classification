from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from aeon.classification.convolution_based import (
     Arsenal,
     MultiRocketClassifier,
     RocketClassifier
 )

from aeon.classification.interval_based import (
    SupervisedTimeSeriesForest
)

def generate_AR1_class1(n, v, delta):
    ts = []  # Initialize the time series list
    x_ini = np.random.normal(0, 1/v)  # Generate x_ini from a uniform distribution
    for i in range(n):
        if i == 0:
            ts.append(2 * delta * np.cos(2 * np.pi * (i / n)) * x_ini + np.random.normal(0, 1 / np.sqrt(v)))  # First value
        else:
            ts.append(2 * delta * np.cos(2 * np.pi * (i / n)) * ts[i - 1] + np.random.normal(0, 1 / np.sqrt(v)))  # Subsequent values
    return ts



def generate_AR1_class2(n, v, delta):
    ts = []  # Initialize the time series list
    x_ini = np.random.normal(0, 1/v)  # Generate x_ini from a uniform distribution
    for i in range(n):
        if i == 0:
            ts.append(delta * np.cos(2 * np.pi * (i / n)) * x_ini + np.random.normal(0, 1 / np.sqrt(v)))  # First value
        else:
            ts.append(delta * np.cos(2 * np.pi * (i / n)) * ts[i - 1] + np.random.normal(0, 1 / np.sqrt(v)))  # Subsequent values
    return ts

# Parameters
n_iterations_class1 = 100  # Training iterations per class
n_iterations_class2 = 100  # Training iterations per class
n_elements = 1024 # Length of each time series
n_iterations_t = 25  # Testing iterations per class
repeats = 500  # Number of times to repeat the process

# Delta values from 0.05 to 0.40
delta_values = np.arange(0.05, 0.45, 0.05)
arr = []
# Add 252 "1"s
for _ in range(n_iterations_class1):
    arr.append('1')
# Add 252 "0"s
for _ in range(n_iterations_class2):
    arr.append('0')
Y_label = np.array(arr, dtype='<U1')

arr = []
# Add 252 "1"s
for _ in range(25):
    arr.append('1')
# Add 252 "0"s
for _ in range(25):
    arr.append('0')
Y_test_label = np.array(arr, dtype='<U1')

# Placeholder for accuracy results
accuracy_means = []
accuracy_stds = []


# Loop over different delta values
for ind_delta in delta_values:
    accuracies = []  # Store accuracies for each repeat

    for i in range(repeats):
        # Generate training and testing datasets
        ts_class1_train = [generate_AR1_class1(n_elements, 1, ind_delta) for _ in range(n_iterations_class1)]
        ts_class2_train = [generate_AR1_class2(n_elements, 1, ind_delta) for _ in range(n_iterations_class2)]
        simu_eps1_train = np.array(ts_class1_train + ts_class2_train).reshape(n_iterations_class1 + n_iterations_class2, 1, n_elements)

        ts_class1_test = [generate_AR1_class1(n_elements, 1, ind_delta) for _ in range(n_iterations_t)]
        ts_class2_test = [generate_AR1_class2(n_elements, 1, ind_delta) for _ in range(n_iterations_t)]
        simu_eps1_test = np.array(ts_class1_test + ts_class2_test).reshape(50, 1, n_elements)
        # Train & predict
        arsenal = SupervisedTimeSeriesForest(n_estimators=200, random_state=47)
        arsenal.fit(simu_eps1_train, Y_label)
        y_pred_arsenal = arsenal.predict(simu_eps1_test)

        # Compute accuracy
        acc = accuracy_score(Y_test_label, y_pred_arsenal)
        accuracies.append(acc)

    # Store mean and std dev for each delta
    accuracy_means.append(np.mean(accuracies))
    accuracy_stds.append(np.std(accuracies))

# Print the accuracy for delta values from 0.05 to 0.40, in order.
print(accuracy_means)
print(accuracy_stds)

