# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Import the libraries.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing training set
dataset_train = pd.read_csv('../Recurrent_Neural_Networks/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

#Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data set with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, training_set_scaled.shape[0]):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
