import numpy as np
import pandas as pd


def MisClassRate(trueLabels, predict):
    """
    Calculate the misclassification rate of the algorithm
    
    Parameters:
    trueLabels   n*1 true labels of each observation  
    predict      n*1 predict labels of each observation 
    
    Returns:     misclassification rate 
    """
    n = trueLabels.shape[0]
    df = pd.DataFrame({'True':trueLabels, 'Predict':predict['Labels'],'V':1})
    table = pd.pivot_table(df, values ='V', index = ['True'], columns=['Predict'], aggfunc=np.sum).fillna(0)
    misRate = 1-sum(table.max(axis=1))/n
    return misRate