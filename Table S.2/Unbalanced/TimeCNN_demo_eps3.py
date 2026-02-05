
from aeon.classification.deep_learning import TimeCNNClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
# from aeon.classification.convolution_based import (
#     Arsenal,
#     MultiRocketClassifier,
#     RocketClassifier
# )
from aeon.classification.interval_based import (
    SupervisedTimeSeriesForest
)
# --- Helper: Epsilon 3 Sigma Calculation ---
def get_sigma_eps3(idx_1_based, n, v):
    """
    Implements the term: (v/2 + v * (i/n) * (1/2))
    idx_1_based corresponds to 'i' in the R code.
    """
    return (v / 2) + (v * (idx_1_based / n) * 0.5)

# ==========================================
# Case 1 (with Epsilon 3)
# ==========================================

def generate_AR1_class1(n, v):
    ts = np.zeros(n)
    x_ini = np.random.normal(0, 1)
    
    for i in range(n):
        t = i + 1 # 1-based index
        sigma = get_sigma_eps3(t, n, v)
        noise = np.random.normal(0, 1)
        
        if t == 1:
            ts[i] = 0.2 * np.cos(2 * np.pi * (t/n)) * x_ini + sigma * noise
        else:
            ts[i] = 0.2 * np.cos(2 * np.pi * (t/n)) * ts[i-1] + sigma * noise
    return ts

def generate_AR1_class2(n, v):
    ts = np.zeros(n)
    x_ini = np.random.normal(0, 1)
    
    for i in range(n):
        t = i + 1
        sigma = get_sigma_eps3(t, n, v)
        noise = np.random.normal(0, 1)
        
        if t == 1:
            ts[i] = 0.4 * np.cos(2 * np.pi * (t/n)) * x_ini + sigma * noise
        else:
            ts[i] = 0.4 * np.cos(2 * np.pi * (t/n)) * ts[i-1] + sigma * noise
    return ts

# ==========================================
# Case 2
# ==========================================

def generate_AR2_class1(n, v):
    ts = np.zeros(n)
    w = np.random.normal(0, 1, n + 2)
    
    for i in range(n):
        t = i + 1
        sigma = get_sigma_eps3(t, n, v)
        
        if t == 1:
            ts[i] = (0.6 * np.sin(2 * np.pi * (t/n))) * w[0] + 0.4 * w[1] + sigma * w[i+2]
        elif t == 2:
            ts[i] = (0.6 * np.sin(2 * np.pi * (t/n))) * w[1] + 0.4 * ts[i-1] + sigma * w[i+2]
        else:
            ts[i] = (0.6 * np.sin(2 * np.pi * (t/n))) * ts[i-2] + 0.4 * ts[i-1] + sigma * w[i+2]
    return ts

def generate_AR2_class2(n, v):
    ts = np.zeros(n)
    w = np.random.normal(0, 1, n + 2)
    
    for i in range(n):
        t = i + 1
        sigma = get_sigma_eps3(t, n, v)
        
        if t == 1:
            ts[i] = 0.4 * np.cos(2 * np.pi * (t/n)) * w[0] + 0.6 * w[1] + sigma * w[i+2]
        elif t == 2:
            ts[i] = 0.4 * np.cos(2 * np.pi * (t/n)) * w[1] + 0.6 * ts[i-1] + sigma * w[i+2]
        else:
            ts[i] = 0.4 * np.cos(2 * np.pi * (t/n)) * ts[i-2] + 0.6 * ts[i-1] + sigma * w[i+2]
    return ts

# ==========================================
# Case 3
# ==========================================

def generate_nAR1_class1(n, v):
    ts = np.zeros(n)
    x_ini = np.random.normal(0, 1)
    
    for i in range(n):
        t = i + 1
        sigma = get_sigma_eps3(t, n, v)
        noise = np.random.normal(0, 1)
        
        if t == 1:
            ts[i] = 0.4 * (np.cos(2 * np.pi * (t/n)) + 1) * x_ini + sigma * noise
        else:
            ts[i] = 0.4 * (np.cos(2 * np.pi * (t/n)) + 1) * ts[i-1] + sigma * noise
    return ts

def generate_MA2_class2(n, v):
    ts = np.zeros(n)
    epsilon = np.random.normal(0, 1, n + 2)
    
    for k in range(n):
        # Logic Mapping:
        # Python ts[k] (time t=k+1) corresponds to R ts[i-2] where R_i = k+3.
        # R terms: epsilon[i-2], epsilon[i-1], epsilon[i]
        # Python indices: epsilon[k], epsilon[k+1], epsilon[k+2]
        
        # Calculate sigmas based on the specific time lags in the R code:
        # Term 1 uses (i-2)/n -> (k+1)/n
        sigma_1 = get_sigma_eps3(k + 1, n, v)
        # Term 2 uses (i-1)/n -> (k+2)/n
        sigma_2 = get_sigma_eps3(k + 2, n, v)
        # Term 3 uses i/n -> (k+3)/n
        sigma_3 = get_sigma_eps3(k + 3, n, v)
        
        e1 = epsilon[k]
        e2 = epsilon[k+1]
        e3 = epsilon[k+2]
        
        ts[k] = 0.3 * sigma_1 * e1 + 0.4 * sigma_2 * e2 + sigma_3 * e3
    return ts

# ==========================================
# Case 4
# ==========================================

def generate_nAR1_2_class1(n, v):
    ts = np.zeros(n)
    x_ini = np.random.normal(0, 1)
    
    for i in range(n):
        t = i + 1
        sigma = get_sigma_eps3(t, n, v)
        noise = np.random.normal(0, 1)
        
        if t == 1:
            ts[i] = (1.5 * np.sin(2 * np.pi * (t/n))) * np.exp(-(t/n) * (x_ini**2)) + sigma * noise
        else:
            ts[i] = (1.5 * np.sin(2 * np.pi * (t/n))) * np.exp(-(t/n) * (ts[i-1]**2)) + sigma * noise
    return ts

def generate_nAR1_2_class2(n, v):
    ts = np.zeros(n)
    x_ini = np.random.normal(0, 1)
    
    for i in range(n):
        t = i + 1
        sigma = get_sigma_eps3(t, n, v)
        noise = np.random.normal(0, 1)
        
        if t == 1:
            ts[i] = (0.5 * np.cos(2 * np.pi * (t/n))) * np.exp(-(t/n) * (-x_ini**2)) + sigma * noise
        else:
            ts[i] = (0.5 * np.cos(2 * np.pi * (t/n))) * np.exp(-(t/n) * (ts[i-1]**2)) + sigma * noise
    return ts

# ==========================================
# Case 5
# ==========================================

def generate_AR2_2_class1(n, v):
    ts = np.zeros(n)
    w = np.random.normal(0, 1, n + 2)
    
    for i in range(n):
        t = i + 1
        sigma = get_sigma_eps3(t, n, v)
        
        if t == 1:
            ts[i] = 0.2 * w[0] + 0.2 * np.sin(2 * np.pi * (t/n)) * w[1] + sigma * w[i+2]
        elif t == 2:
            ts[i] = 0.2 * w[1] + 0.2 * np.sin(2 * np.pi * (t/n)) * ts[i-1] + sigma * w[i+2]
        else:
            ts[i] = 0.2 * ts[i-2] + 0.2 * np.sin(2 * np.pi * (t/n)) * ts[i-1] + sigma * w[i+2]
    return ts

def generate_AR2_2_class2(n, v):
    ts = np.zeros(n)
    w = np.random.normal(0, 1, n + 2)
    
    for i in range(n):
        t = i + 1
        sigma = get_sigma_eps3(t, n, v)
        
        if t == 1:
            ts[i] = (0.2 * np.sin(2 * np.pi * (t/n))) * w[0] + 0.2 * w[1] + sigma * w[i+2]
        elif t == 2:
            ts[i] = (0.2 * np.sin(2 * np.pi * (t/n))) * w[1] + 0.2 * ts[i-1] + sigma * w[i+2]
        else:
            ts[i] = (0.2 * np.sin(2 * np.pi * (t/n))) * ts[i-2] + 0.2 * ts[i-1] + sigma * w[i+2]
    return ts

# ==========================================
# Case 6
# ==========================================

def generate_nAR2_class1(n, v):
    ts = np.zeros(n)
    # Important: R code divides w by v here
    w = np.random.normal(0, 1, n + 2) / v
    
    for i in range(n):
        t = i + 1
        sigma = get_sigma_eps3(t, n, v)
        
        if t == 1:
            ts[i] = 0.2 * np.exp(-(t/n) * (w[0]**2)) + 0.2 * (np.sin(2*np.pi*(t/n))+1) * (1/(w[1]+1)) + sigma * w[i+2]
        elif t == 2:
            ts[i] = 0.2 * np.exp(-(t/n) * (w[1]**2)) + 0.2 * (np.sin(2*np.pi*(t/n))+1) * (1/(ts[i-1]+1)) + sigma * w[i+2]
        else:
            ts[i] = 0.2 * np.exp(-(t/n) * (ts[i-2]**2)) + 0.2 * (np.sin(2*np.pi*(t/n))+1) * (1/(ts[i-1]+1)) + sigma * w[i+2]
    return ts

def generate_nAR2_class2(n, v):
    ts = np.zeros(n)
    # Important: R code divides w by v here
    w = np.random.normal(0, 1, n + 2) / v
    
    for i in range(n):
        t = i + 1
        sigma = get_sigma_eps3(t, n, v)
        
        if t == 1:
            ts[i] = 0.3 * (np.sin(2*np.pi*(t/n))+1) * w[0] + 0.2 * np.exp(-(t/n)*(w[1]**2)) + sigma * w[i+2]
        elif t == 2:
            ts[i] = 0.3 * (np.sin(2*np.pi*(t/n))+1) * w[1] + 0.2 * np.exp(-(t/n)*(ts[i-1]**2)) + sigma * w[i+2]
        else:
            ts[i] = 0.3 * (np.sin(2*np.pi*(t/n))+1) * ts[i-2] + 0.2 * np.exp(-(t/n)*(ts[i-1]**2)) + sigma * w[i+2]
    return ts


# 2. Define the list of cases
# Format: (Case Name, Class 1 Generator, Class 2 Generator)
cases = [
    ("Case 1 (AR1)",       generate_AR1_class1,     generate_AR1_class2),
    ("Case 2 (AR2)",       generate_AR2_class1,     generate_AR2_class2),
    ("Case 3 (nAR1/MA2)",  generate_nAR1_class1,    generate_MA2_class2),
    ("Case 4 (nAR1.2)",    generate_nAR1_2_class1,  generate_nAR1_2_class2),
    ("Case 5 (AR2.2)",     generate_AR2_2_class1,   generate_AR2_2_class2),
    ("Case 6 (nAR2)",      generate_nAR2_class1,    generate_nAR2_class2),
]

# Parameters
n_iterations_class1 = 50  # Training iterations per class
n_iterations_class2 = 250  # Training iterations per class
n_elements = 1024   # Length of each time series
n_iterations_t = 25  # Testing iterations per class
repeats = 500       # Number of times to repeat the process

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

# Train the model.
all_results = {} # Initialize storage

for case_name, gen_func_1, gen_func_2 in cases:

    simunb_eps1 = []
    for i in range(repeats): # Repeat the process 500 times
        ts_class1_train = [gen_func_1(n_elements, 1) for _ in range(n_iterations_class1)]
        ts_class2_train = [gen_func_2(n_elements, 1) for _ in range(n_iterations_class2)]
        simu_eps1_train = np.array(ts_class1_train + ts_class2_train).reshape(n_iterations_class1+n_iterations_class2, 1, n_elements)
        
        ts_class1_test = [gen_func_1(n_elements, 1) for _ in range(n_iterations_t)]
        ts_class2_test = [gen_func_2(n_elements, 1) for _ in range(n_iterations_t)]
        simu_eps1_test = np.array(ts_class1_test + ts_class2_test).reshape(50, 1, n_elements)
         # Store the generated data for this iteration
        simunb_eps1.append({
        "train": simu_eps1_train,
        "test": simu_eps1_test
    })
        
    accuracies = []
    for idx in range(repeats):
        eps1_train = simunb_eps1[idx]['train']
        eps1_test = simunb_eps1[idx]['test']
        cnn = TimeCNNClassifier(n_epochs = 20)
        cnn.fit(eps1_train, Y_label)
        y_pred_STSForest = cnn.predict(eps1_test)
        acc = accuracy_score(Y_test_label, y_pred_STSForest)
        accuracies.append(acc)


    print(np.mean(accuracies))
    print(np.std(accuracies)) 

    # Store results
    all_results[case_name] = {
        "mean": np.mean(accuracies),
        "std": np.std(accuracies)
    }

# The output is inorder for Model 1 to Model 6
print(all_results)