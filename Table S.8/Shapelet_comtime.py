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

from time import time
import time

arr = []
# Add 252 "1"s
for _ in range(252):
    arr.append('1')
# Add 252 "0"s
for _ in range(252):
    arr.append('0')
Y_label = np.array(arr, dtype='<U1')

arr = []
# Add 252 "1"s
for _ in range(75):
    arr.append('1')
# Add 252 "0"s
for _ in range(75):
    arr.append('0')
Y_test_label = np.array(arr, dtype='<U1')


prefixes = ["CZ", "C3"]
indices = [1, 2, 4, 6, 8, 10, 12]

for prefix in prefixes:
    for idx in indices:
        # 1. Construct File Paths
        dataset_name = f"{prefix}_{idx}"
        file_train = f"/home/chenqian/train_{dataset_name}.csv"
        file_test = f"/home/chenqian/test_{dataset_name}.csv"
        
        print("-" * 50)
        print(f"Running for dataset: {dataset_name}")
        print(f"Reading: {file_train}")

        # 2. Load Data
        train_C = pd.read_csv(file_train)
        test_C = pd.read_csv(file_test)

        # 3. Preprocessing (Your specific logic)
        tr_C = train_C.values
        tr_C = tr_C[:, np.newaxis, :]  # Shape: (N, 1, D)
        
        test_C = test_C.values
        test_C = test_C[:, np.newaxis, :] # Shape: (N, 1, D)


        # 4. Remove first column (as per your snippet)
        test_C = test_C[:, :, 1:]
        tr_C = tr_C[:, :, 1:]
        
        # 5. Run Algorithm & Timing
        print("Starting Arsenal...")
        start = time.process_time()
        
        afc = ShapeletTransformClassifier(estimator=RandomForestClassifier(ccp_alpha=0.01), n_shapelet_samples = 10)
        
        # Note: Ensure 'Y_label' is defined/updated for this specific dataset loop!
        afc.fit(tr_C, Y_label) 
        y_pred_Arsenal = afc.predict(test_C)
        # Calculate Accuracy
        acc = accuracy_score(Y_test_label, y_pred_Arsenal)
        end = time.process_time()
        print(f"CPU time: {end - start:.6f} seconds")