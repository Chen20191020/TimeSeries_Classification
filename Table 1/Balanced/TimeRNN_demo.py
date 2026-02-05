from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
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


# -------------------------
# Case 2
# -------------------------

def generate_AR2_class1(n, v):
    ts = np.zeros(n)
    w = np.random.normal(0, 1/v, n+2)
    for i in range(n):
        t = (i+1) / n
        if i == 0:
            ts[i] = 0.6*np.sin(2*np.pi*t) * w[0] + 0.4*w[1] + w[i+2]
        elif i == 1:
            ts[i] = 0.6*np.sin(2*np.pi*t) * w[1] + 0.4*ts[i-1] + w[i+2]
        else:
            ts[i] = 0.6*np.sin(2*np.pi*t) * ts[i-2] + 0.4*ts[i-1] + w[i+2]
    return ts

def generate_AR2_class2(n, v):
    ts = np.zeros(n)
    w = np.random.normal(0, 1/v, n+2)
    for i in range(n):
        t = (i+1) / n
        if i == 0:
            ts[i] = 0.4*np.cos(2*np.pi*t) * w[0] + 0.6*w[1] + w[i+2]
        elif i == 1:
            ts[i] = 0.4*np.cos(2*np.pi*t) * w[1] + 0.6*ts[i-1] + w[i+2]
        else:
            ts[i] = 0.4*np.cos(2*np.pi*t) * ts[i-2] + 0.6*ts[i-1] + w[i+2]
    return ts


# -------------------------
# Case 3
# -------------------------

def generate_nAR1_class1(n, v):
    ts = np.zeros(n)
    x_ini = np.random.normal(0, 1/v)
    for i in range(n):
        t = (i+1)/n
        if i == 0:
            ts[i] = 0.4*(np.cos(2*np.pi*t)+1) * x_ini + np.random.normal(0, 1/v)
        else:
            ts[i] = 0.4*(np.cos(2*np.pi*t)+1) * ts[i-1] + np.random.normal(0, 1/v)
    return ts


def generate_MA2_class2(n, v):
    ts = np.zeros(n)
    eps = np.random.normal(0, 1/v, n+2)
    for i in range(2, n+2):
        ts[i-2] = 0.3*eps[i-2] + 0.4*eps[i-1] + eps[i]
    return ts


# -------------------------
# Case 4
# -------------------------

def generate_nAR1_2_class1(n, v):
    ts = np.zeros(n)
    x_ini = np.random.normal(0, 1/v)
    for i in range(n):
        t = (i+1)/n
        if i == 0:
            ts[i] = 1.5*np.sin(2*np.pi*t)*np.exp(-t * x_ini**2) + np.random.normal(0, 1/v)
        else:
            ts[i] = 1.5*np.sin(2*np.pi*t)*np.exp(-t * ts[i-1]**2) + np.random.normal(0, 1/v)
    return ts

def generate_nAR1_2_class2(n, v):
    ts = np.zeros(n)
    x_ini = np.random.normal(0, 1/v)
    for i in range(n):
        t = (i+1)/n
        if i == 0:
            ts[i] = 0.5*np.cos(2*np.pi*t)*np.exp(-t * (-x_ini**2)) + np.random.normal(0, 1/v)
        else:
            ts[i] = 0.5*np.cos(2*np.pi*t)*np.exp(-t * ts[i-1]**2) + np.random.normal(0, 1/v)
    return ts


# -------------------------
# Case 5
# -------------------------


def generate_AR2_2_class1(n, v):
    ts = np.zeros(n)
    w = np.random.normal(0, 1/v, n+2)
    for i in range(n):
        t = (i+1)/n
        if i == 0:
            ts[i] = 0.2*w[0] + 0.2*np.sin(2*np.pi*t)*w[1] + w[i+2]
        elif i == 1:
            ts[i] = 0.2*w[1] + 0.2*np.sin(2*np.pi*t)*ts[i-1] + w[i+2]
        else:
            ts[i] = 0.2*ts[i-2] + 0.2*np.sin(2*np.pi*t)*ts[i-1] + w[i+2]
    return ts


def generate_AR2_2_class2(n, v):
    ts = np.zeros(n)
    w = np.random.normal(0, 1/v, n+2)
    for i in range(n):
        t = (i+1)/n
        if i == 0:
            ts[i] = 0.2*np.sin(2*np.pi*t)*w[0] + 0.2*w[1] + w[i+2]
        elif i == 1:
            ts[i] = 0.2*np.sin(2*np.pi*t)*w[1] + 0.2*ts[i-1] + w[i+2]
        else:
            ts[i] = 0.2*np.sin(2*np.pi*t)*ts[i-2] + 0.2*ts[i-1] + w[i+2]
    return ts




# -------------------------
# Case 6
# -------------------------

def generate_nAR2_class1(n, v):
    ts = np.zeros(n)
    w = np.random.normal(0, 1/v, n+2)
    for i in range(n):
        t = (i+1)/n
        if i == 0:
            ts[i] = 0.2*np.exp(-t*w[0]**2) + 0.2*(np.sin(2*np.pi*t)+1)*(1/(w[1]+1)) + w[i+2]
        elif i == 1:
            ts[i] = 0.2*np.exp(-t*w[1]**2) + 0.2*(np.sin(2*np.pi*t)+1)*(1/(ts[i-1]+1)) + w[i+2]
        else:
            ts[i] = 0.2*np.exp(-t*ts[i-2]**2) + 0.2*(np.sin(2*np.pi*t)+1)*(1/(ts[i-1]+1)) + w[i+2]
    return ts

def generate_nAR2_class2(n, v):
    ts = np.zeros(n)
    w = np.random.normal(0, 1/v, n+2)
    for i in range(n):
        t = (i+1)/n
        if i == 0:
            ts[i] = 0.3*(np.sin(2*np.pi*t)+1)*w[0] + 0.2*np.exp(-t*w[1]**2) + w[i+2]
        elif i == 1:
            ts[i] = 0.3*(np.sin(2*np.pi*t)+1)*w[1] + 0.2*np.exp(-t*ts[i-1]**2) + w[i+2]
        else:
            ts[i] = 0.3*(np.sin(2*np.pi*t)+1)*ts[i-2] + 0.2*np.exp(-t*ts[i-1]**2) + w[i+2]
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
n_iterations_class1 = 100  # Training iterations per class
n_iterations_class2 = 100  # Training iterations per class
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
        # Train the model.
        History = model.fit(eps1_train, Y_label,
                    epochs=EPOCH,
                    batch_size=BATCH,
                    validation_split=0.0,
                    validation_data=(eps1_test[:M_TEST], Y_test_label[:M_TEST]),
                    shuffle=True,verbose=0,
                    callbacks=[lr_decay])
        pred_RNN = (model.predict(eps1_test) > 0.5).astype(int)
        acc = accuracy_score(Y_test_label, pred_RNN)
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

