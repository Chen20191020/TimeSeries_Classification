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

from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2

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


Y_label = Y_label.astype('float32')
Y_test_label = Y_test_label.astype('float32')

# Let's make a list of CONSTANTS for modelling:
LAYERS = [20, 20, 20, 1]                # number of units in hidden and output layers
M_TRAIN = n_iterations_class1 + n_iterations_class2          # number of training examples (2D)
M_TEST = 50             # number of test examples (2D),full=X_test.shape[0]
N = n_elements              # number of features
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
        History = model.fit(simu_eps1_train, Y_label,
                    epochs=EPOCH,
                    batch_size=BATCH,
                    validation_split=0.0,
                    validation_data=(simu_eps1_test[:M_TEST], Y_test_label[:M_TEST]),
                    shuffle=True,verbose=0,
                    callbacks=[lr_decay])
        pred_RNN = (model.predict(simu_eps1_test) > 0.5).astype(int)
        acc = accuracy_score(Y_test_label, pred_RNN)
        accuracies.append(acc)

    # Store mean and std dev for each delta
    accuracy_means.append(np.mean(accuracies))
    accuracy_stds.append(np.std(accuracies))

# Print the accuracy for delta values from 0.05 to 0.40, in order.
print(accuracy_means)
print(accuracy_stds)

