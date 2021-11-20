#!/usr/bin/env python
# coding: utf-8

# # 2A

# In[11]:


#Function to generate an identity matrix of size n
def identity_matrix(n):
    a = [[0 for y in range(n)] for x in range(n)]
    for i in range(n):
        a[i][i] = 1
    return a


# In[12]:


n = 5
A = identity_matrix(n)
print(A," is the identity matrix")


# # 2B

# In[13]:


# Changing third column of a to 2
for i in range(n):
    A[i][2] = 2
print(A, " after changing third column to 2")


# # 2C

# In[14]:


#Determining sum of all elements in the matrix
total_sum = 0
for x in A:
    total_sum += sum(x)

print("sum of all the elements in the matrix ", total_sum)


# # 2D

# In[15]:


#Matrix transposition
A = [[A[j][i] for j in range(n)] for i in range(n)]

print(A, " is the transposed matrix")


# # 2E

# In[28]:


#Determining sum of third row, sum of the diagonal and sum of the second column in the matrix
sum_of_third_row = sum(A[2][i] for i in range(n))

print(sum_of_third_row)

sum_of_diag = sum(A[i][i] for i in range(n))

print(sum_of_diag)

sum_of_second_col = sum(A[i][1] for i in range(n))

print(sum_of_second_col)


# # 2F

# In[17]:


# Generating a standard normal matrix B. A standard normal matrix is a matrix with mean value 0 and standard deviation 1
import numpy as np
np.random.seed(42)
B = np.random.standard_normal(size = (5,5))

print(B)


# # 2G

# In[18]:


# Generating matrix C
C = np.zeros((2,5))
C[0] = B[0] - A[1]
C[1] = A[3] + B[4]
print(C, " is the resultant matrix")


# # 2H

# In[29]:


D = [C[:,i] * (i+2) for i in range(n)]
print(D)


# # 2I

# In[20]:


# X = [1; 3; 5; 7]T , Y = [4; 3; 2; 1]T , Z = [2; 4; 6; 8]T. Find covariance matrix of X,Y,Z
X = np.array([1,3,5,7])
Y = np.array([4,3,2,1])
Z = np.array([2,4,6,8])

print(np.cov([X,Y,Z]))

# Pearson correlation co-efficient of X and Y
np.corrcoef(X,Y)


# # 2J

# In[21]:


import statistics
data = [20, 1, 3, 5, 7, 9, 14]
data1 = [x**2 for x in data]

m1 = statistics.mean(data)
m2 = statistics.mean(data1)
sd1 = statistics.pstdev(data)
sd2 = statistics.stdev(data)
var1 = sd1 ** 2
var2 = sd2 ** 2

#LHS 
print('LHS Mean is : ', m2)
#RHS for Population standard deviation
print('RHS Mean for Population Standard Deviation',m1**2 + var1)
#RHS for Sampling standard deviation
print('RHS Mean for Sampling Standard Deviation',m1**2 + var2)

