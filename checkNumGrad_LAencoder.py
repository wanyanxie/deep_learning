import numpy as np
from scipy.sparse import *
def checkNumgrad (epsilon,data_set,model) :
    """ Gradient checking """
    theta=model.theta
    epsilon_vector = np.array(np.zeros(np.size(theta)))
    numGrad = np.array(np.zeros(np.size(theta)))
    for p in range (np.size(theta)):
        print p
        epsilon_vector[p] = epsilon
        J1_grad=model.linear_autoencoder_Cost_Grad(theta+epsilon_vector,data_set)
        J2_grad=model.linear_autoencoder_Cost_Grad(theta-epsilon_vector,data_set)
        numGrad[p] = (J1_grad[0]-J2_grad[0])/(2*epsilon)
        epsilon_vector[p] = 0    
    return numGrad, J1_grad[1] 