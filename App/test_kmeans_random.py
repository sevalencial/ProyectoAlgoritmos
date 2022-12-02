# script for testing the algorithms

import numpy as np
import matplotlib.pyplot as plt
from utils.kmeans_func import KMeans
from utils.data_gen import get_random_data

k = 20
n = 10000
d = 15

data = get_random_data(k,n,d)

centroids_initial = data[np.random.choice(range(data.shape[0]), k, replace=False),:]
output_k = KMeans(data, k, centroids_initial)


## plot the first two dimensions
centroids = data[np.random.choice(range(data.shape[0]), k, replace=False),:]

cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0, 1, k)]

centroids1 =output_k["Centroids"]
labels1 = output_k["Labels"]

for i,color in enumerate(colors,start =1):
    plt.scatter(data[labels1==i, :][:,0], data[labels1==i, :][:,1], color=color)

for j in range(k):
    plt.scatter(centroids1[j,0],centroids1[j,1],color = 'w',marker='x') 

plt.show()