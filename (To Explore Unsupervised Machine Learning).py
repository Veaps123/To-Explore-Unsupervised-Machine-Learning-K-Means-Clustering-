#!/usr/bin/env python
# coding: utf-8

# In[71]:


import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("C:\\Users\\hp\\Desktop\\Iris.csv")
data


# In[73]:


data.head()


# In[10]:


data.info()


# In[11]:


data.describe().all


# In[74]:


data.drop(['Id'],axis=1)
data.head()


# In[75]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

plt.scatter(data.SepalLengthCm , data.SepalWidthCm, data.PetalLengthCm , data.PetalWidthCm)


# In[76]:


km = KMeans(n_clusters = 3)
km


# In[77]:


X = data.iloc[:, [0, 1, 2, 3]].values
y = data['Species'].values


# In[78]:


level = LabelEncoder()
y = level.fit_transform(y)


# In[79]:


fig, ax = plt.subplots(figsize=(15,6))
plt.subplot(1,2,1)
sns.scatterplot(x = data.SepalLengthCm, y =  data.SepalWidthCm,
                hue = data.Species, style = data.Species, palette=['black', 'green', 'red']);


# In[80]:


fig, ax = plt.subplots(figsize=(15,6))
plt.subplot(1,2,2)
sns.scatterplot(x = data.PetalLengthCm, y =  data.PetalWidthCm,
                hue = data.Species, style = data.Species, palette=['black', 'green', 'red']);


# In[81]:


plt.subplot(2,2,1)
plt.hist(data['SepalLengthCm'], rwidth=0.9)
plt.title('Sepal Length')

plt.subplot(2,2,2)
plt.hist(data['SepalWidthCm'],rwidth=0.9)
plt.title('Sepal Width')

plt.subplot(2,2,3)
plt.hist(data['PetalLengthCm'],rwidth=0.9)
plt.xlabel('Petal Length')

plt.subplot(2,2,4)
plt.hist(data['PetalWidthCm'],rwidth=0.9)
plt.xlabel('Petel Width')


# In[82]:


import seaborn as sns
plt.subplot(2,2,1)
sns.kdeplot(data['SepalLengthCm'])
plt.title('Sepal Length')

plt.subplot(2,2,2)
sns.kdeplot(data['SepalWidthCm'])
plt.title('Sepal Width')

plt.subplot(2,2,3)
sns.kdeplot(data['PetalLengthCm'])
plt.xlabel('Petal Length')

plt.subplot(2,2,4)
sns.kdeplot(data['PetalWidthCm'])
plt.xlabel('Petel Width')


# In[83]:


dist_points_from_cluster_center = []
K = range(1,10)
# n = no_of_clusters
for n in K:
    
    kmc = KMeans(n_clusters=n)
    kmc.fit(X)
    dist_points_from_cluster_center.append(kmc.inertia_)


# In[84]:


plt.plot(K,dist_points_from_cluster_center)


# In[85]:


# Applying kmeans to the dataset / Creating the kmeans classifier
kmc = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_pred = kmc.fit_predict(X)


# In[86]:


plt.scatter(X[y_pred == 0, 0],  X[y_pred == 0, 1], s= 50, c= 'red',label='Iris-setosa')
plt.scatter(X[y_pred == 1, 0],  X[y_pred == 1, 1], s= 50, c = 'blue',label='Iris-versicolour')
plt.scatter(X[y_pred == 2, 0],  X[y_pred == 2, 1], s= 50, c = 'green',label='Iris-virginica')

plt.scatter(kmc.cluster_centers_[:,0], kmc.cluster_centers_[:,1], s= 100, c = 'Purple',label='centroids')
plt.title('Clusters of Iris Dataset')
plt.legend()
plt.show()


# In[ ]:


Thank youvery much

