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

from sklearn.ensemble import RandomForestClassifier
from aeon.classification.shapelet_based import ShapeletTransformClassifier


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
n_iterations_t = 25  # Testing iterations per class
repeats = 500  # Number of times to repeat the process
delta_val = 0.2
# Delta values from 500 to 1500
n_values = np.arange(500, 1600, 100)
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
for ind_n in n_values:
    accuracies = []  # Store accuracies for each repeat
    n_elements = ind_n
    for i in range(repeats):
        # Generate training and testing datasets
        ts_class1_train = [generate_AR1_class1(n_elements, 1, delta_val) for _ in range(n_iterations_class1)]
        ts_class2_train = [generate_AR1_class2(n_elements, 1, delta_val) for _ in range(n_iterations_class2)]
        simu_eps1_train = np.array(ts_class1_train + ts_class2_train).reshape(n_iterations_class1 + n_iterations_class2, 1, n_elements)

        ts_class1_test = [generate_AR1_class1(n_elements, 1, delta_val) for _ in range(n_iterations_t)]
        ts_class2_test = [generate_AR1_class2(n_elements, 1, delta_val) for _ in range(n_iterations_t)]
        simu_eps1_test = np.array(ts_class1_test + ts_class2_test).reshape(50, 1, n_elements)

        stc = Arsenal()

        stc.fit(simu_eps1_train, Y_label)
        y_pred_Shapelets = stc.predict(simu_eps1_test)
        acc = accuracy_score(Y_test_label, y_pred_Shapelets)
        
        accuracies.append(acc)
        
    # Store mean and std dev for each delta
    accuracy_means.append(np.mean(accuracies))
    accuracy_stds.append(np.std(accuracies))

# The accuracy for n values from 500 to 1500, in order.
print(accuracy_means)
print(accuracy_stds)
