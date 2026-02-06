from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from aeon.classification.shapelet_based import ShapeletTransformClassifier
from aeon.visualisation import ShapeletClassifierVisualizer

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
n_iterations_class2 = 100 # Training iterations per class
n_elements = 1024
n_iterations_t = 25  # Testing iterations per class
ts_class1_train = [generate_AR1_class1(n_elements, 1) for _ in range(n_iterations_class1)]
ts_class2_train = [generate_AR1_class2(n_elements, 1) for _ in range(n_iterations_class2)]
simu_train = np.array(ts_class1_train + ts_class2_train).reshape(n_iterations_class1+n_iterations_class2, 1, n_elements)

# Generate testing data for Model 1
ts_class1_test = [generate_AR1_class1(n_elements, 1) for _ in range(n_iterations_t)]
ts_class2_test = [generate_AR1_class2(n_elements, 1) for _ in range(n_iterations_t)]
simu_test = np.array(ts_class1_test + ts_class2_test).reshape(50, 1, n_elements)

arr = []
# Add 252 "1"s
for _ in range(25):
    arr.append('1')
# Add 252 "0"s
for _ in range(25):
    arr.append('0')
Y_test_label = np.array(arr, dtype='<U1')

arr = []
    # Add 252 "1"s
for _ in range(n_iterations_class1):
      arr.append('1')
for _ in range(n_iterations_class2):
      arr.append('0')
Y_label = np.array(arr, dtype='<U1')

Y_label = Y_label.astype('float32')
Y_test_label = Y_test_label.astype('float32')

# simu_train = simu_train.reshape(simu_train.shape[0], simu_train.shape[2])
stc = ShapeletTransformClassifier(
    estimator=RandomForestClassifier(ccp_alpha=0.01), n_shapelet_samples=10
)
stc.fit(simu_train, Y_label)
id_class = 0
stc_vis = ShapeletClassifierVisualizer(stc)

fig = stc_vis.visualize_shapelets_one_class(
    simu_test,
    Y_test_label,
    id_class,
    figure_options={"figsize": (10, 6), "nrows": 2, "ncols": 2},
)
# 1. Access the hidden transformer inside the classifier
# This object HAS the .transform() method you were looking for
transformer = stc._transformer

# 2. Calculate the distances (Min Distance) for ALL shapelets
# Output shape: (n_samples, n_shapelets)
all_distances = transformer.transform(simu_test)

# 3. Get the data for the FIRST shapelet (Row 1, Col 1 of your plot)
shapelet_index = 0
shapelet_0_dists = all_distances[:, shapelet_index]

# 4. Split by Class (to separate the Green and Orange box data)
# (Assuming 'id_class' is the class you focused on, e.g., Class 0)
orange_box_data = shapelet_0_dists[Y_test_label == id_class]
green_box_data  = shapelet_0_dists[Y_test_label != id_class]

# feature for boxplot
feature = np.concatenate([green_box_data, orange_box_data])
print(feature)