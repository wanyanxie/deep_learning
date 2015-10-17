import cPickle, gzip
import numpy as np
from scipy.sparse import *
import scipy.optimize
import scipy.io

import sys
sys.path.append("/Users/wanyanxie/Summer_2014_Study/Deep_Learning/")
import LinearAutoencoder
reload(LinearAutoencoder)
from checkNumGrad_LAencoder import checkNumgrad
from LinearAutoencoder import linear_autoencoder
import ZCA
reload(ZCA)
from ZCA import Whitening
#from ZCA import check_covariance 


from autoencoder import autoencoder
from showImages import showHiddenIMAGES
from feedForwardAutoencoder import feedForwardAutoencoder 
from softmaxTrain import softmaxTrain
from softmaxPredict import softmaxPredict
from stackedAECost import stackedAE
from checkNumGrad_FineTuning import checkNumgrad
from feedForwardstackedAEC import feedForwardstackedAEC



####### load data
train_set= scipy.io.loadmat('/Users/wanyanxie/Summer_2014_Study/Deep_Learning/stlSampledPatches.mat')['patches']

###  Initialize constants and parameters 
imageChannels=3                                          # number of coloer channels
patch_size=8
input_layer_size = patch_size * patch_size * imageChannels # Size of input vector (MNIST images are 28x28)
numLabels  = 10                                          # Number of classes (MNIST images fall into 10 classes)
lambd = 3e-3                                             # Weight decay parameter 
hidden_layer_size = 400                           # hidden layer 1 Size
                     
rho = 0.035                                              # desired average activation of the hidden units.                        
beta = 5                                               # weight of sparsity penalty term   
epsilon = 0.1;	       #epsilon for ZCA whitening


##### check gradient
#input_layer_size_debug = 8*8*3;
#hidden_layer_size_debug = 5;
#train_set_debug=train_set.T[1:100].T
#LAencoder= linear_autoencoder(input_layer_size_debug, hidden_layer_size_debug, beta, rho, lambd)
#LAencoder.linear_autoencoder_Cost_Grad(LAencoder.theta,train_set_debug)
#epsilon_check=1e-4
#check=checkNumgrad (epsilon_check,train_set_debug,LAencoder)  

###### ZCA-whitening
train_set= Whitening(train_set, epsilon)



#######  Train the second sparse autoencoder
iterations = 400
model=linear_autoencoder(input_layer_size, hidden_layer_size, beta, rho, lambd)
theta= scipy.optimize.minimize(model.linear_autoencoder_Cost_Grad, x0=model.theta, 
                               args = (train_set,), 
                               method = 'L-BFGS-B', 
                               jac = True, 
                               options = {'maxiter': iterations}) 