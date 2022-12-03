from utils.distance_func import distance
from utils.kmeanspp_func import cost

def ClusterCost(data,predict):
    dist = distance(data,predict["Centroids"])
    return cost(dist)/(10**4)