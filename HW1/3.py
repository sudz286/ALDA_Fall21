
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fast')


# In[4]:


df = pd.read_csv('wine.csv')
df.head()


# # 3A

# In[12]:


# Function called for solution for 3.1
def attribute_describe(df):

    print("Mean = ",df.mean(),"Median = ", df.median(),"Standard Deviation =",df.std(),"Range = ",df.max() - df.min())
    print("25th percentile =", df.quantile(q = 0.25),"50th percentile =", df.quantile(q = 0.50),"75th percentile", df.quantile(q = 0.75),end = '\n\n')


# In[13]:


# Solution to 3.1
arr = []
arr.append(df['Alcohol'])
arr.append(df['Malic acid'])
arr.append(df['Ash'])
arr.append(df['Alcalinity of ash'])

for x in arr:
    attribute_describe(x)


# # 3B

# In[18]:


# Solution for 3.2
boxplot = df.boxplot(column =['Ash','Malic acid'],by="Class",figsize = [7,7])
plt.show()

# ## 3C Histogram Proanthocyanins

# In[ ]:


# Solution for 3.3
df1=df['Proanthocyanins']
df1.hist(bins=16)
plt.title("Proanthocyanins")
plt.show()

# ## 3C Proline

# In[ ]:


df2=df['Proline']
df2.hist(bins=16)
plt.title("Proline")
plt.show()

# ## 3D

# In[14]:


df1 = df[['Flavanoids', 'Total phenols','Ash','Malic acid']]
df2 = df['Class']
pd.plotting.scatter_matrix(df1, diagonal = 'kde',alpha = 0.5,figsize = [10,10],c = df2)

plt.show()


# ## 3E

# In[50]:


from mpl_toolkits.mplot3d import Axes3D

plt.style.use('classic')
fig = plt.figure(figsize = [10,10])
ax = fig.add_subplot(projection = '3d')
dfcolor = df['Class']
x=df['Proanthocyanins']
y=df['Flavanoids']
z=df['Total phenols']
ax.scatter(x, y, z,c=dfcolor)
ax.set_xlabel("Proanthocyanins",fontweight="bold")
ax.set_ylabel("Flavanoids",fontweight="bold")
ax.set_zlabel("Total phenols",fontweight="bold")
plt.title("Three Dimensional Scatter Plot",fontweight="bold")
plt.show()


# ## **3F**

# In[42]:


import scipy.stats as stats
def qq_x(x,title):
    z = (x-np.mean(x))/np.std(x)
    stats.probplot(z, dist="norm", plot=plt)
    plt.title(title)
    plt.show()
    
qq_x(df['Ash'],"Q-Q Plot for Ash column")
qq_x(df['OD280/OD315 of diluted wines'], "Q-Q Plot for Diluted Wines Column")