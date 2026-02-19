from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from aeon.classification.convolution_based import (
     Arsenal,
     MultiRocketClassifier,
     RocketClassifier
 )

from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2



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

Y_label = Y_label.astype('float32')
Y_test_label = Y_test_label.astype('float32')



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
        
        # Let's make a list of CONSTANTS for modelling:
        LAYERS = [20, 20, 20, 1]                # number of units in hidden and output layers
        M_TRAIN = tr_C.shape[0]           # number of training examples (2D)
        M_TEST = test_C.shape[0]             # number of test examples (2D),full=X_test.shape[0]
        N = tr_C.shape[2]                 # number of features
        BATCH = M_TRAIN                          # batch size
        EPOCH = 50                           # number of epochs
        LR = 5e-2                            # learning rate of the gradient descent
        LAMBD = 3e-2                         # lambda in L2 regularizaion
        DP = 0.0                             # dropout rate 
        RDP = 0.0                            # recurrent dropout rate
        T = 1                   # Time step windows
        # Build the Model


        model = Sequential()
        model.add(LSTM(input_shape=(T, N), units=LAYERS[0],
               activation='tanh', recurrent_activation='hard_sigmoid',
               kernel_regularizer=l2(LAMBD), recurrent_regularizer=l2(LAMBD),
               dropout=DP, recurrent_dropout=RDP,
               return_sequences=True, return_state=False,
               stateful=False, unroll=False
              ))
        model.add(BatchNormalization())
        model.add(LSTM(units=LAYERS[1],
               activation='tanh', recurrent_activation='hard_sigmoid',
               kernel_regularizer=l2(LAMBD), recurrent_regularizer=l2(LAMBD),
               dropout=DP, recurrent_dropout=RDP,
               return_sequences=True, return_state=False,
               stateful=False, unroll=False
              ))
        model.add(BatchNormalization())
        model.add(LSTM(units=LAYERS[2],
               activation='tanh', recurrent_activation='hard_sigmoid',
               kernel_regularizer=l2(LAMBD), recurrent_regularizer=l2(LAMBD),
               dropout=DP, recurrent_dropout=RDP,
               return_sequences=False, return_state=False,
               stateful=False, unroll=False
              ))
        model.add(BatchNormalization())
        model.add(Dense(units=LAYERS[3], activation='sigmoid'))


        # Compile the model with Adam optimizer
        model.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer=Adam(learning_rate=LR))


        # Define a learning rate decay method:
        lr_decay = ReduceLROnPlateau(monitor='loss',
                             patience=1, verbose=0,
                             factor=0.5, min_lr=1e-8)
        # Define Early Stopping:
        early_stop = EarlyStopping(monitor='accuracy', min_delta=0,
                           patience=30, verbose=1, mode='min',
                           baseline=0, restore_best_weights=True)


        # 5. Run Algorithm & Timing
        print("Starting TimeRNN...")
        start = time.process_time()

        History = model.fit(tr_C, Y_label,
                    epochs=EPOCH,
                    batch_size=BATCH,
                    validation_split=0.0,
                    validation_data=(test_C[:M_TEST], Y_test_label[:M_TEST]),
                    shuffle=True,verbose=0,
                    callbacks=[lr_decay])

        # Evaluate the model:
        pred_RNN_CZ = (model.predict(test_C) > 0.5).astype(int)
        end = time.process_time()
        print(f"CPU time: {end - start:.6f} seconds")