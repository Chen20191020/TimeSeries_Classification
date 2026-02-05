from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from aeon.classification.convolution_based import (
     Arsenal,
     MultiRocketClassifier,
     RocketClassifier
 )

from sklearn.ensemble import RandomForestClassifier
from aeon.classification.shapelet_based import ShapeletTransformClassifier


# Helper for the variance term used repeatedly
def get_sigma(idx_1_based, n, v):
    # Matches the term: (v/4 + v/4*(cos(2*pi*(i/n))^2))
    return v/4 + (v/4 * (np.cos(2 * np.pi * (idx_1_based / n)) ** 2))

# ==========================================
# Case 1 (with Epsilon 2)
# ==========================================

def generate_AR1_class1(n, v):
    ts = np.zeros(n)
    x_ini = np.random.normal(0, 1)
    
    for i in range(n):
        # i is 0-based index, t is 1-based index to match R logic
        t = i + 1 
        sigma = get_sigma(t, n, v)
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
        sigma = get_sigma(t, n, v)
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
        sigma = get_sigma(t, n, v)
        
        # w indexing: R's w[i+2] is Python's w[i+2] due to shift,
        # but R's w[1] is w[0] and w[2] is w[1].
        
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
        sigma = get_sigma(t, n, v)
        
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
        sigma = get_sigma(t, n, v)
        noise = np.random.normal(0, 1)
        
        if t == 1:
            ts[i] = 0.4 * (np.cos(2 * np.pi * (t/n)) + 1) * x_ini + sigma * noise
        else:
            ts[i] = 0.4 * (np.cos(2 * np.pi * (t/n)) + 1) * ts[i-1] + sigma * noise
            
    return ts

def generate_MA2_class2(n, v):
    ts = np.zeros(n)
    epsilon = np.random.normal(0, 1, n + 2)
    
    # R loop: for(i in 3:(n+2)). 
    # Logic: Calculates ts[1] using eps[1], eps[2], eps[3] (indices adjusted for R)
    # Python equivalent: loop k from 0 to n-1
    
    for k in range(n):
        # To match R's 'i' in the formula:
        # When k=0 (first ts element), R's i was 3.
        current_i = k + 3 
        
        term1_idx = current_i - 2 # R index i-2 -> Python i-3?
        # Let's map directly:
        # R epsilon indices: i-2, i-1, i.
        # Python indices (0-based): (current_i-2)-1, (current_i-1)-1, current_i-1
        # Simplifies to: k, k+1, k+2
        
        e1 = epsilon[k]
        e2 = epsilon[k+1]
        e3 = epsilon[k+2]
        
        # Sigma calculations based on R time indices
        sigma_1 = v/4 + (v/4 * (np.cos(2 * np.pi * ((current_i-2)/n)) ** 2))
        sigma_2 = v/4 + (v/4 * (np.cos(2 * np.pi * ((current_i-1)/n)) ** 2))
        sigma_3 = v/4 + (v/4 * (np.cos(2 * np.pi * (current_i/n)) ** 2))
        
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
        sigma = get_sigma(t, n, v)
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
        sigma = get_sigma(t, n, v)
        noise = np.random.normal(0, 1)
        
        if t == 1:
            # Note: R code had -x_ini^2. In R -x^2 is -(x^2). Same in Python.
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
        sigma = get_sigma(t, n, v)
        
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
        sigma = get_sigma(t, n, v)
        
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
    w = np.random.normal(0, 1, n + 2)
    
    for i in range(n):
        t = i + 1
        sigma = get_sigma(t, n, v)
        
        if t == 1:
            ts[i] = 0.2 * np.exp(-(t/n) * (w[0]**2)) + 0.2 * (np.sin(2*np.pi*(t/n))+1) * (1/(w[1]+1)) + sigma * w[i+2]
        elif t == 2:
            ts[i] = 0.2 * np.exp(-(t/n) * (w[1]**2)) + 0.2 * (np.sin(2*np.pi*(t/n))+1) * (1/(ts[i-1]+1)) + sigma * w[i+2]
        else:
            ts[i] = 0.2 * np.exp(-(t/n) * (ts[i-2]**2)) + 0.2 * (np.sin(2*np.pi*(t/n))+1) * (1/(ts[i-1]+1)) + sigma * w[i+2]
            
    return ts

def generate_nAR2_class2(n, v):
    ts = np.zeros(n)
    w = np.random.normal(0, 1, n + 2)
    
    for i in range(n):
        t = i + 1
        sigma = get_sigma(t, n, v)
        
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
# Placeholder to store all generated datasets
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
        stc = ShapeletTransformClassifier(estimator=RandomForestClassifier(ccp_alpha=0.01), n_shapelet_samples = 10)
        stc.fit(eps1_train, Y_label)
        y_pred_multirock = stc.predict(eps1_test)
        acc = accuracy_score(Y_test_label, y_pred_multirock)
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
    




