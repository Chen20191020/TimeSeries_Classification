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
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
from sklearn.ensemble import RandomForestClassifier
from aeon.classification.shapelet_based import ShapeletTransformClassifier
import rpy2.robjects as robjects
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


# Read data function 
def load_and_resample_rds(file_path, target_n=4515):
    """
    Reads an RDS list and resamples every vector to length target_n.
    Returns shape (m, 1, target_n).
    """
    # 1. Read the RDS file
    readRDS = robjects.r['readRDS']
    r_list = readRDS(file_path)
   
    # 2. Convert R objects to a list of numpy arrays
    vectors = [np.array(item) for item in r_list]
    m = len(vectors)

    # 3. Initialize the result array (m, 1, n)
    result_array = np.zeros((m, 1, target_n))

    # 4. Process each vector
    for i, vec in enumerate(vectors):
        vec = vec.flatten() # Ensure it's 1D
        current_len = vec.size
       
        if current_len > 0:
            # --- The Resampling Logic ---
            # Create a grid of points from 0 to 1 for the original data
            original_x = np.linspace(0, 1, current_len)
           
            # Create a grid of points from 0 to 1 for the target data (n points)
            target_x = np.linspace(0, 1, target_n)
           
            # Interpolate: Calculate what the values would be at the target points
            resampled_vec = np.interp(target_x, original_x, vec)
           
            # Store in the matrix
            result_array[i, 0, :] = resampled_vec
           
    return result_array

# Read data
tr_C3 = load_and_resample_rds("C:/Users/chenq/Desktop/train_set/train_C3.rds")
test_C3 =  load_and_resample_rds("C:/Users/chenq/Desktop/test_set/test_C3.rds")

tr_C4 = load_and_resample_rds("C:/Users/chenq/Desktop/train_set/train_C4.rds")
test_C4 =  load_and_resample_rds("C:/Users/chenq/Desktop/test_set/test_C4.rds")

tr_CZ = load_and_resample_rds("C:/Users/chenq/Desktop/train_set/train_CZ.rds")
test_CZ =  load_and_resample_rds("C:/Users/chenq/Desktop/test_set/test_CZ.rds")


# Classes
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

## TimeCNN
# C3 Channel 
cnn = TimeCNNClassifier(n_epochs = 20)
cnn.fit(tr_C3, Y_label)
y_pred_timeCNN_C3 = cnn.predict(test_C3)
accuracy_score(Y_test_label, y_pred_timeCNN_C3)

# C4 Channel 
cnn = TimeCNNClassifier(n_epochs = 20)
cnn.fit(tr_C4, Y_label)
y_pred_timeCNN_C4 = cnn.predict(test_C4)
accuracy_score(Y_test_label, y_pred_timeCNN_C4)

# CZ Channel 
cnn = TimeCNNClassifier(n_epochs = 20)
cnn.fit(tr_CZ, Y_label)
y_pred_timeCNN_CZ = cnn.predict(test_CZ)
print(accuracy_score(Y_test_label, y_pred_timeCNN_CZ))

# Majority voting 
y_pred_timeCNN_CZ = y_pred_timeCNN_CZ.astype('float32')
y_pred_timeCNN_C3 = y_pred_timeCNN_C3.astype('float32')
y_pred_timeCNN_C4 = y_pred_timeCNN_C4.astype('float32')
# Stack the arrays and calculate the majority for each element
stacked_arrays = np.stack([y_pred_timeCNN_CZ, y_pred_timeCNN_C3, y_pred_timeCNN_C4], axis=0)
majority_timeCNN = np.sum(stacked_arrays, axis=0) > 1  # More than half will be 1
print(accuracy_score(Y_test_label.astype(int), majority_timeCNN))


## Rocket
# C3 Channel 
rocket = RocketClassifier()
rocket.fit(tr_C3, Y_label)
y_pred_Rocket_C3 = rocket.predict(test_C3)
print(accuracy_score(Y_test_label, y_pred_Rocket_C3))

# C4 Channel 
rocket = RocketClassifier()
rocket.fit(tr_C4, Y_label)
y_pred_Rocket_C4 = rocket.predict(test_C4)
print(accuracy_score(Y_test_label, y_pred_Rocket_C4))

# CZ Channel 
rocket = RocketClassifier()
rocket.fit(tr_CZ, Y_label)
y_pred_Rocket_CZ = rocket.predict(test_CZ)
print(accuracy_score(Y_test_label, y_pred_Rocket_CZ))

# Majority voting 
y_pred_Rocket_CZ = y_pred_Rocket_CZ.astype('float32')
y_pred_Rocket_C3 = y_pred_Rocket_C3.astype('float32')
y_pred_Rocket_C4 = y_pred_Rocket_C4.astype('float32')
# Stack the arrays and calculate the majority for each element
stacked_arrays = np.stack([y_pred_Rocket_CZ, y_pred_Rocket_C3, y_pred_Rocket_C4], axis=0)
majority_Rocket = np.sum(stacked_arrays, axis=0) > 1  # More than half will be 1
print(accuracy_score(Y_test_label.astype(int), majority_Rocket))


## Arsenal
# C3 Channel 
afc = Arsenal()
afc.fit(tr_C3, Y_label)
y_pred_Arsenal_C3 = afc.predict(test_C3)
print(accuracy_score(Y_test_label, y_pred_Arsenal_C3))

# C4 Channel 
afc = Arsenal()
afc.fit(tr_C4, Y_label)
y_pred_Arsenal_C4 = afc.predict(test_C4)
print(accuracy_score(Y_test_label, y_pred_Arsenal_C4))

# CZ Channel 
afc = Arsenal()
afc.fit(tr_CZ, Y_label)
y_pred_Arsenal_CZ = afc.predict(test_CZ)
print(accuracy_score(Y_test_label, y_pred_Arsenal_CZ))

# Majority voting 
y_pred_Arsenal_CZ = y_pred_Arsenal_CZ.astype('float32')
y_pred_Arsenal_C3 = y_pred_Arsenal_C3.astype('float32')
y_pred_Arsenal_C4 = y_pred_Arsenal_C4.astype('float32')
# Stack the arrays and calculate the majority for each element
stacked_arrays = np.stack([y_pred_Arsenal_CZ, y_pred_Arsenal_C3, y_pred_Arsenal_C4], axis=0)
majority_Arsenal = np.sum(stacked_arrays, axis=0) > 1  # More than half will be 1
print(accuracy_score(Y_test_label.astype(int), majority_Arsenal))


## MultiRocket
# C3 Channel 
multi_arsenal = Arsenal(rocket_transform="multirocket")
multi_arsenal.fit(tr_C3, Y_label)
y_pred_multirock_C3 = multi_arsenal.predict(test_C3)
print(accuracy_score(Y_test_label, y_pred_multirock_C3))

# C4 Channel 
multi_arsenal = Arsenal(rocket_transform="multirocket")
multi_arsenal.fit(tr_C4, Y_label)
y_pred_multirock_C4 = multi_arsenal.predict(test_C4)
print(accuracy_score(Y_test_label, y_pred_multirock_C4))

# CZ Channel 
multi_arsenal = Arsenal(rocket_transform="multirocket")
multi_arsenal.fit(tr_CZ, Y_label)
y_pred_multirock_CZ = multi_arsenal.predict(test_CZ)
print(accuracy_score(Y_test_label, y_pred_multirock_CZ))

# Majority voting 
y_pred_multirock_CZ = y_pred_multirock_CZ.astype('float32')
y_pred_multirock_C3 = y_pred_multirock_C3.astype('float32')
y_pred_multirock_C4 = y_pred_multirock_C4.astype('float32')
# Stack the arrays and calculate the majority for each element
stacked_arrays = np.stack([y_pred_multirock_CZ, y_pred_multirock_C3, y_pred_multirock_C4], axis=0)
majority_multirock = np.sum(stacked_arrays, axis=0) > 1  # More than half will be 1
print(accuracy_score(Y_test_label.astype(int), majority_multirock))


## TimeRNN
# Let's make a list of CONSTANTS for modelling:
LAYERS = [20, 20, 20, 1]                # number of units in hidden and output layers
M_TRAIN = tr_CZ.shape[0]           # number of training examples (2D)
M_TEST = test_CZ.shape[0]             # number of test examples (2D),full=X_test.shape[0]
N = tr_CZ.shape[2]                 # number of features
BATCH = M_TRAIN                          # batch size
EPOCH = 50                           # number of epochs
LR = 5e-2                            # learning rate of the gradient descent
LAMBD = 3e-2                         # lambda in L2 regularizaion
DP = 0.0                             # dropout rate
RDP = 0.0                            # recurrent dropout rate
T = 1                               # Time step windows

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


# C3 Channel 
# Train the model. 
History = model.fit(tr_C3, Y_label,
                    epochs=EPOCH,
                    batch_size=BATCH,
                    validation_split=0.0,
                    validation_data=(test_C3[:M_TEST], Y_test_label[:M_TEST]),
                    shuffle=True,verbose=0,
                    callbacks=[lr_decay])

# Evaluate the model:
pred_RNN_C3 = (model.predict(test_C3) > 0.5).astype(int)
print(accuracy_score(Y_test_label, pred_RNN_C3))

# C4 Channel 
# Train the model. 
History = model.fit(tr_C4, Y_label,
                    epochs=EPOCH,
                    batch_size=BATCH,
                    validation_split=0.0,
                    validation_data=(test_C4[:M_TEST], Y_test_label[:M_TEST]),
                    shuffle=True,verbose=0,
                    callbacks=[lr_decay])

# Evaluate the model:
pred_RNN_C4 = (model.predict(test_C4) > 0.5).astype(int)
print(accuracy_score(Y_test_label, pred_RNN_C4))

# CZ Channel 
# Train the model. 
History = model.fit(tr_CZ, Y_label,
                    epochs=EPOCH,
                    batch_size=BATCH,
                    validation_split=0.0,
                    validation_data=(test_CZ[:M_TEST], Y_test_label[:M_TEST]),
                    shuffle=True,verbose=0,
                    callbacks=[lr_decay])

# Evaluate the model:
pred_RNN_CZ = (model.predict(test_CZ) > 0.5).astype(int)
print(accuracy_score(Y_test_label, pred_RNN_CZ))

# Majority voting 
pred_RNN_CZ = pred_RNN_CZ.astype('float32')
pred_RNN_C3 = pred_RNN_C3.astype('float32')
pred_RNN_C4 = pred_RNN_C4.astype('float32')
# Stack the arrays and calculate the majority for each element
stacked_arrays = np.stack([pred_RNN_CZ, pred_RNN_C3, pred_RNN_C4], axis=0)
majority_rnn = np.sum(stacked_arrays, axis=0) > 1  # More than half will be 1
print(accuracy_score(Y_test_label.astype(int), majority_rnn))

## Shapelets 
# C3 Channel 
stc = ShapeletTransformClassifier(
    estimator=RandomForestClassifier(ccp_alpha=0.01), n_shapelet_samples=10
)
stc.fit(tr_C3, Y_label)
y_pred_Shapelets_C3 = stc.predict(test_C3)
print(stc.score(test_C3, Y_test_label))

# C4 Channel 
stc = ShapeletTransformClassifier(
    estimator=RandomForestClassifier(ccp_alpha=0.01), n_shapelet_samples=10
)
stc.fit(tr_C4, Y_label)
y_pred_Shapelets_C4 = stc.predict(test_C4)
print(stc.score(test_C4, Y_test_label))

# CZ Channel 
stc = ShapeletTransformClassifier(
    estimator=RandomForestClassifier(ccp_alpha=0.01), n_shapelet_samples=10
)
stc.fit(tr_CZ, Y_label)
y_pred_Shapelets_CZ = stc.predict(test_CZ)
print(stc.score(test_CZ, Y_test_label))

# Majority voting 
y_pred_Shapelets_CZ = y_pred_Shapelets_CZ.astype('float32')
y_pred_Shapelets_C3 = y_pred_Shapelets_C3.astype('float32')
y_pred_Shapelets_C4 = y_pred_Shapelets_C4.astype('float32')
# Stack the arrays and calculate the majority for each element
stacked_arrays = np.stack([y_pred_Shapelets_CZ, y_pred_Shapelets_C3, y_pred_Shapelets_C4], axis=0)
majority_Shapelets = np.sum(stacked_arrays, axis=0) > 1  # More than half will be 1
accuracy_score(Y_test_label.astype(int), majority_Shapelets)

## STSForest
# C3 Channel 
stsf = SupervisedTimeSeriesForest(n_estimators=50, random_state=47)
stsf.fit(tr_C3, Y_label)
y_pred_STSForest_C3 = stsf.predict(test_C3)
print("STSF Accuracy: " + str(metrics.accuracy_score(Y_test_label, y_pred_STSForest_C3)))

# C4 Channel 
stsf = SupervisedTimeSeriesForest(n_estimators=50, random_state=47)
stsf.fit(tr_C4, Y_label)
y_pred_STSForest_C4 = stsf.predict(test_C4)
print("STSF Accuracy: " + str(metrics.accuracy_score(Y_test_label, y_pred_STSForest_C4)))

# CZ Channel 
stsf = SupervisedTimeSeriesForest(n_estimators=200, random_state=47)
stsf.fit(tr_CZ, Y_label)
y_pred_STSForest_CZ = stsf.predict(test_CZ)
print("STSF Accuracy: " + str(metrics.accuracy_score(Y_test_label, y_pred_STSForest_CZ)))

# Majority voting 
y_pred_STSForest_CZ = y_pred_STSForest_CZ.astype('float32')
y_pred_STSForest_C3 = y_pred_STSForest_C3.astype('float32')
y_pred_STSForest_C4 = y_pred_STSForest_C4.astype('float32')
# Stack the arrays and calculate the majority for each element
stacked_arrays = np.stack([y_pred_STSForest_CZ, y_pred_STSForest_C3, y_pred_STSForest_C4], axis=0)
majority_STSForest = np.sum(stacked_arrays, axis=0) > 1  # More than half will be 1
accuracy_score(Y_test_label.astype(int), majority_STSForest)