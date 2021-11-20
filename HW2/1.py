
import pandas as pd
import numpy as np

# Loading Training and Testing datasets
train = pd.read_csv(r'data\pca_train.csv')
test = pd.read_csv(r'data\pca_test.csv')
train.head()

print('Shape of training and testing datasets',train.shape, test.shape)
print('Size of training and testing datasets',train.shape, test.shape)

print('Class 0 samples in Training set')
print(len(train[train['Class'] == 0]))
print('Class 1 samples in Training Set')
print(len(train[train['Class'] == 1]))

print('Number of Class 0 samples in Testing set', len(test[test['Class'] == 0]))
print('Number of Class 1 samples in Testing Set', len(test[test['Class'] == 1]))

columns = train.columns[:-1]
y_train = train['Class']
y_test =  test['Class']

# Normalizing Training and Testing data sets
from sklearn import preprocessing
import pandas as pd

for c in columns:
  max = train[c].max()
  min = train[c].min()
  test[c] = test[c].apply(lambda x: (x-min)/(max-min))
  train[c] = train[c].apply(lambda x: (x-min)/(max-min))

train = train[columns]
test = test[columns]
train.head()

def covar(x):
    #Calculating covariance matrix of the training dataset
    return x.cov()
    # DDOF = 0 gives us the population covariance
    
cov_train = covar(train)
cov_test = covar(test)
print('Dimensions of the Training covariance matrix is', cov_train.shape)
print("First 5 rows and columns of the covariance matrix")
cov_train[cov_train.columns[:5]].head(5)

from numpy.linalg import eig

eigenvalues, eigenvectors = eig(cov_train)
print("Size of covariance matrix is ", cov_train.size)
print("5 Largest eigenvalues are ")
print(sorted(eigenvalues, reverse=True)[:5])

import matplotlib.pyplot as plt
def plot_eigen(eigen_matrix):
    fig = plt.figure()
    fig.set_figwidth(20)
    fig.set_figheight(10)
    plt.xlabel('Number of Eigenvalues')
    plt.ylabel('Eigenvalue')
    plt.plot(eigenvalues)
    plt.show()

plot_eigen(train)

# Determing the percentage of variance accounted for by each of the first 30 components
df = pd.DataFrame(columns, columns=['Attribute'])
df['eigenvalues'] = eigenvalues

df['eigen percentage'] = df['eigenvalues']/sum(df['eigenvalues'])
df['cummulative eigenvalue percent'] = np.cumsum(df['eigen percentage'])

df

plt.figure(figsize=(20,10))
plt.plot(range(len(eigenvalues)), df['cummulative eigenvalue percent'] * 100)
plt.xlabel("Number of components")
plt.ylabel("Cumulative eigenvalues")
plt.show()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

p_values = [2,4,8,10,20,25,30]
accuracy_list = []
for p in p_values:
  pca_train = train.dot(eigenvectors[:,:p])
  pca_test = test.dot(eigenvectors[:,:p])
  knn = KNeighborsClassifier(5).fit(pca_train, y_train)
  y_pred = knn.predict(pca_test)
  accuracy_list.append(accuracy_score(y_test, y_pred))
  if p == 10:
    test_report = pca_test
    test_report.columns = test.columns[:10]
    test_report['Ground Truth Output'] = y_test
    test_report['Actual Output'] = y_pred
    test_report.to_csv(r'test_output_norm.csv', index = False)
  print("Accuracy of KNN when p = ", p, "->", accuracy_list[len(accuracy_list) - 1])

# Plotting Number of Principal Components vs Accuracy of the KNN Model
plt.figure(figsize=(20,10))
plt.plot(p_values, accuracy_list)
plt.ylabel("Accuracy")
plt.xlabel("Number of Principal Components")
plt.title("Number of Principal Components vs Accuracy of KNN Model")
plt.show()

# Performing Standardization of the data
train = pd.read_csv(r'data\pca_train.csv')
test = pd.read_csv(r'data\pca_test.csv')

y_train = train['Class']
y_test = test['Class']

for c in columns:
  mean = train[c].mean()
  std = np.std(train[c])
  test[c] = test[c].apply(lambda x: (x-mean)/std)
  train[c] = train[c].apply(lambda x: (x-mean)/std)

train = train[columns]
test = test[columns]
train.head()

def covar(x):
    #Calculating covariance matrix of the training dataset
    return x.cov()
    # DDOF = 0 gives us the population covariance
    
cov_train = covar(train)
cov_test = covar(test)
print('Dimensions of the Training covariance matrix is', cov_train.shape)
print("Size of covariance matrix is ", cov_train.size)
print("First 5 rows and columns of the covariance matrix")
cov_train[cov_train.columns[:5]].head(5)

from numpy.linalg import eig

eigenvalues, eigenvectors = eig(cov_train)

print("5 Largest eigenvalues are ")
print(sorted(eigenvalues, reverse=True)[:5])

import matplotlib.pyplot as plt
def plot_eigen(eigen_matrix):
    fig = plt.figure()
    fig.set_figwidth(20)
    fig.set_figheight(10)
    plt.xlabel('Number of Eigenvalues')
    plt.ylabel('Eigenvalue')
    plt.plot(eigenvalues)
    plt.show()

plot_eigen(train)

# Determing the percentage of variance accounted for by each of the first 30 components

import numpy as np

df = pd.DataFrame(columns, columns=['Attribute'])

df['eigenvalues'] = eigenvalues

df['eigen percentage'] = df['eigenvalues']/sum(df['eigenvalues'])
df['cummulative eigenvalue percent'] = np.cumsum(df['eigen percentage'])

df

plt.figure(figsize=(20,10))
plt.plot(range(len(eigenvalues)), df['cummulative eigenvalue percent'] * 100)
plt.xlabel("Number of components")
plt.ylabel("Cumulative eigenvalues")
plt.title("Cumulative eigenvalues vs Number of components")
plt.show()

p_values = [2,4,8,10,20,25,30]
accuracy_list = []
for p in p_values:
  pca_train = train.dot(eigenvectors[:,:p])
  pca_test = test.dot(eigenvectors[:,:p])
  knn = KNeighborsClassifier(5).fit(pca_train, y_train)
  y_pred = knn.predict(pca_test)
  accuracy_list.append(accuracy_score(y_test, y_pred))
  if p == 10:
    test_report = pca_test
    test_report.columns = test.columns[:10]
    test_report['Ground Truth Output'] = y_test
    test_report['Actual Output'] = y_pred
    test_report.to_csv(r'test_output_std.csv', index = False)
  print("Accuracy of KNN when p = ", p, "->", accuracy_list[len(accuracy_list) - 1])

# Plotting Number of Principal Components vs Accuracy of the KNN Model
plt.figure(figsize=(20,10))
plt.plot(p_values, accuracy_list)
plt.ylabel("Accuracy")
plt.xlabel("Number of Principal Components")
plt.title("Number of Principal Components vs Accuracy of KNN Model")
plt.show()
