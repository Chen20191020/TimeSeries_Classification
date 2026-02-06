from sklearn.metrics import accuracy_score
#from aeon.classification.deep_learning import TimeCNNClassifier
import pandas as pd
import numpy as np
# from aeon.classification.convolution_based import (
#     Arsenal,
#     MultiRocketClassifier,
#     RocketClassifier,
# )
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2

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


# Let's make a list of CONSTANTS for modelling:
LAYERS = [20, 20, 20, 1]                # number of units in hidden and output layers
M_TRAIN = simu_eps1_train.shape[0]           # number of training examples (2D)
M_TEST = simu_eps1_test.shape[0]             # number of test examples (2D),full=X_test.shape[0]
N = simu_eps1_train.shape[2]                 # number of features
BATCH = M_TRAIN                          # batch size
EPOCH = 50                           # number of epochs
LR = 5e-2                            # learning rate of the gradient descent
LAMBD = 3e-2                         # lambda in L2 regularizaion
DP = 0.0                             # dropout rate
RDP = 0.0                            # recurrent dropout rate
T = 1               # Time step windows

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


# Train the model.

accuracies = []

# Repeat the training and testing process for each dataset
for idx in range(repeats):
    # Retrieve training and testing datasets
    simu_model1_eps1_train = simunb_eps1[idx]['train']
    simu_model1_eps1_test = simunb_eps1[idx]['test']

    History = model.fit(simu_model1_eps1_train, Y_label,
                    epochs=EPOCH,
                    batch_size=BATCH,
                    validation_split=0.0,
                    validation_data=(simu_model1_eps1_test[:M_TEST], Y_test_label[:M_TEST]),
                    shuffle=True,verbose=0,
                    callbacks=[lr_decay])

    pred_RNN = model.predict(simu_model1_eps1_test)

# the Feature in the boxplot
print(pred_RNN)