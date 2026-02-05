import pyreadr
from sklearn.metrics import accuracy_score
from aeon.classification.deep_learning import TimeCNNClassifier
from aeon.datasets import load_basic_motions  # multivariate dataset
from aeon.datasets import load_italy_power_demand  # univariate dataset
import pandas as pd
import numpy as np
import pickle 
from aeon.classification.convolution_based import (
    Arsenal,
    HydraClassifier,
    MiniRocketClassifier,
    MultiRocketClassifier,
    MultiRocketHydraClassifier,
    RocketClassifier,
)

from sklearn import metrics

from aeon.classification.interval_based import (
    RSTSF,
    CanonicalIntervalForestClassifier,
    DrCIFClassifier,
    QUANTClassifier,
    RandomIntervalSpectralEnsembleClassifier,
    SupervisedTimeSeriesForest,
    TimeSeriesForestClassifier,
)

from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2

from sklearn.ensemble import RandomForestClassifier
from aeon.classification.shapelet_based import ShapeletTransformClassifier
import rpy2.robjects as robjects
from rpy2.robjects import conversion, default_converter

# 1. Load the file
readRDS = robjects.r['readRDS']
r_data = readRDS('/home/chenqian/lsw_model3_unbalance.rds')

# Placeholder to store all generated datasets
lsw_model = []

# Repeat the process 500 times
for i in range(len(r_data)):
    item_1 = r_data[i]   #  Access the first item (Python uses 0-based indexing)
    train_part = item_1.rx2('train') #  Access 'train' inside that element
    x_raw = np.array(train_part[0]) # Access 'x' inside 'train'
    x_train = x_raw.reshape(300, 1, 1024)
    test_part = item_1.rx2('test')
    x_raw = np.array(test_part[0])
    x_test = x_raw.reshape(50, 1, 1024)
    
    # Store the generated data for this iteration
    lsw_model.append({
        "train": x_train,
        "test": x_test
    })

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



# TimeCNN 
accuracies = []

# Repeat the training and testing process for each dataset
for idx in range(len(r_data)):
    # Retrieve training and testing datasets
    eps1_train = lsw_model[idx]['train']
    eps1_test = lsw_model[idx]['test']

    cnn = TimeCNNClassifier(n_epochs=20)
    cnn.fit(eps1_train, Y_label)
    y_pred = cnn.predict(eps1_test)
    acc = accuracy_score(Y_test_label, y_pred)
    accuracies.append(acc)

print(np.mean(accuracies))
print(np.std(accuracies)) 




# Arsenal
accuracies = []

# Repeat the training and testing process for each dataset
for idx in range(500):
    # Retrieve training and testing datasets
    eps1_train = lsw_model[idx]['train']
    eps1_test = lsw_model[idx]['test']
    afc = Arsenal()
    afc.fit(eps1_train, Y_label)
    y_pred = afc.predict(eps1_test)
    acc = accuracy_score(Y_test_label, y_pred)
    accuracies.append(acc)

print(np.mean(accuracies))
print(np.std(accuracies))  


# MultiRocket 

accuracies = []

# Repeat the training and testing process for each dataset
for idx in range(500):
    # Retrieve training and testing datasets
    eps1_train = lsw_model[idx]['train']
    eps1_test = lsw_model[idx]['test']
    afc = Arsenal(rocket_transform="multirocket")
    afc.fit(eps1_train, Y_label)
    y_pred = afc.predict(eps1_test)
    acc = accuracy_score(Y_test_label, y_pred)
    accuracies.append(acc)

print(np.mean(accuracies))
print(np.std(accuracies))    


# Rocket

accuracies = []

# Repeat the training and testing process for each dataset
for idx in range(500):
    # Retrieve training and testing datasets
    eps1_train = lsw_model[idx]['train']
    eps1_test = lsw_model[idx]['test']
    afc = RocketClassifier()
    afc.fit(eps1_train, Y_label)
    y_pred = afc.predict(eps1_test)
    acc = accuracy_score(Y_test_label, y_pred)
    accuracies.append(acc)

print(np.mean(accuracies))
print(np.std(accuracies))  



# Shapelet 
accuracies = []

# Repeat the training and testing process for each dataset
for idx in range(500):
    # Retrieve training and testing datasets
    eps1_train = lsw_model[idx]['train']
    eps1_test = lsw_model[idx]['test']
    afc = ShapeletTransformClassifier(
    estimator=RandomForestClassifier(ccp_alpha=0.01), n_shapelet_samples=10)
    afc.fit(eps1_train, Y_label)
    y_pred = afc.predict(eps1_test)
    acc = accuracy_score(Y_test_label, y_pred)
    accuracies.append(acc)

print(np.mean(accuracies))
print(np.std(accuracies))   


# STSforest 
accuracies = []

# Repeat the training and testing process for each dataset
for idx in range(500):
    # Retrieve training and testing datasets
    eps1_train = lsw_model[idx]['train']
    eps1_test = lsw_model[idx]['test']
    afc = SupervisedTimeSeriesForest(n_estimators=200, random_state=47)
    afc.fit(eps1_train, Y_label)
    y_pred = afc.predict(eps1_test)
    acc = accuracy_score(Y_test_label, y_pred)
    accuracies.append(acc)
    print(idx)
print(np.mean(accuracies))
print(np.std(accuracies))   



# TimeRNN 

Y_label = Y_label.astype('float32')
Y_test_label = Y_test_label.astype('float32')
# Let's make a list of CONSTANTS for modelling:
LAYERS = [20, 20, 20, 1]                # number of units in hidden and output layers
M_TRAIN = 300           # number of training examples (2D)
M_TEST = 50            # number of test examples (2D),full=X_test.shape[0]
N = 1024                # number of features
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

accuracies = []

# Repeat the training and testing process for each dataset
for idx in range(500):
    eps1_train = lsw_model[idx]['train']
    eps1_test = lsw_model[idx]['test']
    History = model.fit(eps1_train, Y_label,
                    epochs=EPOCH,
                    batch_size=BATCH,
                    validation_split=0.0,
                    validation_data=(eps1_test[:M_TEST], Y_test_label[:M_TEST]),
                    shuffle=True,verbose=0,
                    callbacks=[lr_decay])
    pred_RNN_CZ = (model.predict(eps1_test) > 0.5).astype(int)
    acc = accuracy_score(Y_test_label, pred_RNN_CZ)
    accuracies.append(acc)

print(np.mean(accuracies))
print(np.std(accuracies))   