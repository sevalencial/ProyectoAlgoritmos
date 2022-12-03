from utils.kmeans_func import KMeans
from utils.kmeanspp_func import KMeansPlusPlus
from utils.data_gen import get_random_data
import numpy as np
import matplotlib.pyplot as plt

k = 20
n = 10000
d = 15

data = get_random_data(k,n,d)

data = data['data']
centroids_initial = KMeansPlusPlus(data, 20)
output_kpp = KMeans(data, k, centroids_initial)

cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0, 1, k)]

centroids1 =output_kpp["Centroids"]
labels1 = output_kpp["Labels"]
iterations = output_kpp["Iteration before Coverge"]

for i,color in enumerate(colors,start =1):
    plt.scatter(data[labels1==i, :][:,0], data[labels1==i, :][:,1], color=color)

for j in range(k):
    plt.scatter(centroids1[j,0],centroids1[j,1],color = 'w',marker='x')

plt.show()