# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 23:24:45 2021

@author: prady
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 23:42:43 2021

@author: prady
"""
from sklearn.model_selection import LeaveOneOut
import numpy as np
import pandas as pd
import math


def mean_squared(y_test,predicted):
    return np.square(np.subtract(y_test,predicted)).mean()

def linearreg(x, y):
  x_new = np.hstack((np.ones([x.shape[0],1], x.dtype), x))
  x_new_2 = np.dot(np.transpose(x_new),x_new)
  coef = np.dot(np.dot(np.linalg.inv(x_new_2),np.transpose(x_new)),y)
  return coef
  
def predict(x, coef):
  x_new = np.hstack((np.ones([x.shape[0],1], x.dtype), x))
  return np.dot(x_new, coef)

def loocv_rmse(x, y):
  loo = LeaveOneOut()
  rmse = 0
  for train_index, test_index in loo.split(x):
      
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    coef = linearreg(x_train, y_train)
    predicted = predict(x_test, coef)
    mse = mean_squared(y_test, predicted)
    rmse = rmse + mse

  return (rmse/x.shape[0])**0.5

dataset = pd.read_csv('CSC422522_HW4/generator_temperature.csv')
x = np.array(dataset.iloc[:,0:3])
y = np.array(dataset.iloc[:,3])

x2 = np.power(x, 2)
x3 = np.power(x, 3)

print("Coefficients are: \n")

print("For Model (1) - ", linearreg(x,y))

print("For Model (2) - ", linearreg(x2,y))

print("For Model (3) - ", linearreg(x3,y))

print("\nRMSE Values are: \n")

print("Leave-one-out RMSE for Model  (1) - ", loocv_rmse(x, y))

print("Leave-one-out RMSE for Model  (2) - ", loocv_rmse(x2, y))

print("Leave-one-out RMSE for Model  (3) - ", loocv_rmse(x3, y))