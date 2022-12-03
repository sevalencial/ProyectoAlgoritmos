import numpy as np


def get_random_data(k=20, n=10000, d=15):
    """Generate synthetic data sampled from a d-dimensional spherical gaussian distribution with at the origin

    Args:
        k (int): amount of centers to sample
        n (int): size of the dataset
        d (int): amount of dimensions of the multivariate gaussian

    Returns:
        np.ndarray: numpy array containing the data
    """

    ## simulate k centers from 15-dimensional spherical Gaussian distribution 
    mean = np.hstack(np.zeros((d,1)))
    diag = np.linspace(1,d,d)
    diag = 2**diag
    cov = np.diag(diag)
    centers = np.random.multivariate_normal(mean, cov, k)

    ## Simulate n data
    for i in range(k):
        mean = centers[i]
        if i == 0:
            data = np.random.multivariate_normal(mean, np.diag(np.ones(d)), int(n/k+n%k))
            trueLabels = np.repeat(i,int(n/k+n%k))
        else:
            data = np.append(data, np.random.multivariate_normal(mean, np.diag(np.ones(d)) , int(n/k)), axis = 0) 
            trueLabels = np.append(trueLabels,np.repeat(i,int(n/k)))
    
    output = {'data':data, 'trueLabels':trueLabels}
    return output


