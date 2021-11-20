# -*- coding: utf-8 -*-
"""HW4_4 .ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11xQu2l17LK0-mLeHBrE678pQXBNdTxGw
"""

# Answer for question number 4
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np
import random

train = pd.read_csv('train_data_2021.csv')
test = pd.read_csv('test_data_2021.csv')
val_data = pd.read_csv('val_data_2021.csv')

val_dataset=(val_data.loc[:, val_data.columns != 'Class'],val_data['Class'])
train.head()

test.head()

print(train.size,test.size,val_data.size)
print(train.shape,test.shape,val_data.shape)
val_data.head()

import os
os.environ['PYTHONHASHSEED'] = '2021'
random.seed(2021)
np.random.seed(2021)
tf.random.set_seed(2021)

X = [4,16,32,64]
models = []
model_history = []
# We store the size of the hidden neurons in a list X, the correponding model for each neuron size in models and its metric details in model_history

for x in X:
    print("Number of hidden neurons is ",x)
    # Here we define a Sequential model with `relu` activation function for the hidden layer. `signmoid` function for the output layer
    model = Sequential([
        Dense(units=x, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])
    # We then compile the model with the following parameters and fit the model with the validation data set included.
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(x=train.loc[:, train.columns != 'Class'], y=train['Class'],validation_data=val_dataset, batch_size=10, epochs=5, verbose=2)
    # We append the accuracy metrics to the lists model_history and the model to models. 
    models.append(model)
    model_history.append(history)

model_metrics = []
val_metrics = []

for i in model_history:
    model_metrics.append(i.history['accuracy'][-1])
    val_metrics.append(i.history['val_accuracy'][-1])

# inputs = tf.keras.Input(shape=(60,))
# x = [ tf.keras.layers.Dense(hidden_neuron_size, activation= 'relu')(inputs) for hidden_neuron_size in X]
# outputs = [tf.keras.layers.Dense(1, activation='sigmoid')(hidden_layer) for hidden_layer in x]
# models = [ tf.keras.Sequential(inputs=inputs, outputs=output) for output in outputs]

# for model in models:
  # model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])
  # model_metrics.append(model.fit(train.loc[:, train.columns != 'Class'], train['Class'], batch_size= 10,epochs=5))
  # val_metrics.append(model.evaluate(val_data.loc[:, val_data.columns != 'Class'], val_data['Class'], return_dict=True))

import matplotlib.pyplot as plt


plt.plot(X,model_metrics)
plt.scatter(X,model_metrics)
plt.plot(X,val_metrics)
plt.scatter(X,val_metrics)
plt.title('Number of Neurons vs Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of Hidden Neurons')
plt.legend(['Train','Validation'])
plt.show()

test_acc = []
for model in models:
  test_acc.append(model.evaluate(test.loc[:, test.columns != 'Class'], test['Class']))

print("Optimal number of neurons in the network is", X[2], "with accuracy", test_acc[2][1])

import platform
print(platform.python_version())

# Language - Python V3.7.12
# Frameworks used - Tenforflow API V2.0, Matplotlib V3.2.2, numpy V1.19.5 and Pandas V1.1.5
# We've utilised Tensorflow's API to build, train and test our neural network. 
# We've used python's package manager 'pip' V21.1.3 to install the dependencies mentioned above. 

# Steps followed (Assuming you have python and pip installed in your system):
# 1.Change to the directory you wish to work in.
# 2.Run `pip install tensorflow numpy pandas matplotlib` in the command line.
# 3.Store the datasets in the same directory and create a python file with the name of your choice. 
# 4.Run the python file using the command `python3 (insert-file-name().py` and record the output.