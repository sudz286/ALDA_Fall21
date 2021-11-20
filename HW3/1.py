# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 20:14:05 2021

@author: prady
"""
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from numpy import absolute
from numpy import mean
from sklearn.model_selection import GroupKFold
from scipy.spatial.distance import pdist
import pandas as pd
d = { 'ID': [1,2,3,4,5,6,7,8,9,10,11,12], 
      'x1': [-5.86,-10.97,0.79,-0.59,3.63,2.02,-6.41,6.13,-2.35,2.66,-3.71,2.4],
      'x2': [-2.0,-1.0,-2.0,1.0,-2.0,-5.0,-1.0,-7.0,6.0,-3.0,2.0,1.0],
      'y': [0,0,1,1,1,1,0,1,0,1,0,0]}
df = pd.DataFrame(data=d)
X_data = df[['x1','x2']]
Y_data = df['y']

#-----------------------------------------------1(a)----------------------------------------------#

cv1 = LeaveOneOut()

knn1 = KNeighborsClassifier(n_neighbors=1,metric="manhattan")

MAE = cross_val_score(knn1,X_data,Y_data,scoring="neg_mean_absolute_error",cv=cv1, n_jobs=1)

print("The leave-one-out cross-validation error Mean absolute error is ",mean(absolute(MAE)),"\n")

#-----------------------------------------------1(a)----------------------------------------------#


#-----------------------------------------------1(b)----------------------------------------------#

def get_my_key(obj):
    return obj['distance']

def sorting_dist(P):
  dist = [ {'index': i + 1, 'distance': pdist ( [P, [x,y]], metric = 'cityblock'), 'coords': (x,y)} for i,(x,y) in enumerate(zip(X_data['x1'],X_data['x2']))]
  dist.sort(key = get_my_key)
  return dist

nearest_pts_3 = sorting_dist([X_data['x1'][2],X_data['x2'][2]])
nearest_pts_10 = sorting_dist([X_data['x1'][9],X_data['x2'][9]])

print("For data point 3 the 3 closest neighbours respectively are data points:")
for i in range(1,4):
  print('Data Point:' ,nearest_pts_3[i]['index'],'with Distance',nearest_pts_3[i]['distance'])

print("For data point 10 the 3 closest neighbours respectively are data points:") 
for i in range(1,4):
  print('Data Point:' ,nearest_pts_10[i]['index'],'Distance',nearest_pts_10[i]['distance'])
    
#-----------------------------------------------1(b)----------------------------------------------#


#-----------------------------------------------1(c)----------------------------------------------#

groups = []

print("\nBased on the condition specified, all 12 data points are grouped and placed in respective folds - ",end='')

for i in range(1,len(X_data)+1,1):
  groups.append(i%3+1)

print(groups)  

knn3 = KNeighborsClassifier(n_neighbors=4,metric="manhattan")

gf = GroupKFold(n_splits=3)

kf_error = cross_val_score(knn3,X_data,Y_data,cv=gf,scoring="neg_mean_absolute_error",groups=groups)

print("3-folded cross validation error for 3NN would be: ",mean(absolute(kf_error)))

#-----------------------------------------------1(c)----------------------------------------------#