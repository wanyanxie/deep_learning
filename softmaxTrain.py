import numpy as np
import scipy.optimize

import sys
sys.path.append("/Users/wanyanxie/Summer_2014_Study/Deep_Learning/")
from softmaxClass import softmax

def softmaxTrain (inputSize, numClasses, lambd, data_set, option):
    
    """ Initialize the autoencoder with the  parameters above  """
    model=softmax(numClasses, inputSize, lambd)
    """ Training using  L-BFGS algorithm  """
    theta= scipy.optimize.minimize(model.softmax_Cost_Grad, x0=model.theta, 
                                   args = (data_set,), 
                                   method = 'L-BFGS-B', 
                                   jac = True, 
                                   options = option) 
    return theta.x.reshape(numClasses, inputSize)