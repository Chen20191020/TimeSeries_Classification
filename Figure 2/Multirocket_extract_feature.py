import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from aeon.datasets import load_basic_motions  # multivariate dataset
from aeon.datasets import load_gunpoint  # univariate dataset
from aeon.transformations.collection.convolution_based import Rocket
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.classification.convolution_based import Arsenal

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



# fit 
mrocket = MultiRocket() 
mrocket.fit(simu_train)
X_train_transform = mrocket.transform(simu_train)

classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
classifier.fit(X_train_transform, Y_label)

# Feature used in boxplot
X_test_transform = mrocket.transform(simu_test)
classifier.decision_function(X_test_transform)