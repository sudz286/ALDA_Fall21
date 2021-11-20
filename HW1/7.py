#!/usr/bin/env python
# coding: utf-8

# # 7A

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_fwf("auto-mpg.data")
df.columns = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin','car name']
df.head()


# In[2]:


# Displacement vs Weight data and its subsequent plot
dis = df['displacement']
weight = df['weight']

f = plt.figure()
f.set_figwidth(8)
f.set_figheight(8)

plt.scatter(dis,weight)
plt.title("Displacement and Weight Data")
plt.xlabel("Displacment")
plt.ylabel("Weight")


plt.show()

# We can infere that displcement is alomost linearly proportional to the weight of the car barring a few outliers


# # 7B & 7C

# In[4]:


from scipy.spatial.distance import cdist,pdist
def get_my_key(obj):
    return obj['distance']

def six_closest_points(P,df,text):
    dist = [ {'distance': pdist( [P, [x,y]], metric = text), 'coords': (x,y)} for x,y in zip(df['displacement'],df['weight'])]
    dist.sort(key = get_my_key)
    print("Closest 6 points for ",text, " distance are")
    for i in range(6):
        print(dist[i])
    
    coord = [ x['coords'] for x in dist ]
    return coord[:20]
     
        

#Data point P
P = [df['displacement'].mean(), df['weight'].mean()]
print(P)
euclidean_coords = six_closest_points(P,df,'euclidean')
cityblock_coords = six_closest_points(P,df,'cityblock')
minkowski_coords = six_closest_points(P,df,'minkowski')
chebyshev_coords = six_closest_points(P,df,'chebyshev')
cosine_coords = six_closest_points(P,df,'cosine')


# # 7D

# In[6]:


def plot_20_nearest_pts(x,text):
    plt.scatter(*zip(*x))
    plt.title(text)
    plt.xlabel('Displacement')
    plt.ylabel('Weight')
    plt.scatter(P[0],P[1])
    plt.show()

plot_20_nearest_pts(euclidean_coords,'Euclidean distance')
plot_20_nearest_pts(cityblock_coords,'Manhattan distance')
plot_20_nearest_pts(minkowski_coords,'Minkowski distance')
plot_20_nearest_pts(chebyshev_coords,'Chebyshev distance')
plot_20_nearest_pts(cosine_coords,'Cosine distance')

    # We can infer that the nearest 20 points using cosine distance varies significantly from the rest.
    # The Euclidean distance determines the shortest straight line distance between the two points
    # The Cosine Distance however determines the angular distance a line drawn from the origin to the point in consideration makes with the x-axis. The points are then grouped together based on closeness of angle and not conventional straight line distance
    # The distance measured above shows us the cos inverse of the dot product 

