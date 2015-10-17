import cPickle, gzip
import numpy as np
from scipy.sparse import *
import scipy.optimize

import sys
sys.path.append("/Users/wanyanxie/Summer_2014_Study/Deep_Learning/")
from autoencoder import autoencoder
from showImages import showHiddenIMAGES
from feedForwardAutoencoder import feedForwardAutoencoder 
from softmaxTrain import softmaxTrain
from softmaxPredict import softmaxPredict

###  Initialize constants and parameters 
patch_size=28
input_layer_size = patch_size * patch_size               # Size of input vector (MNIST images are 28x28)
numLabels  = 5                                           # Number of classes (MNIST images fall into 10 classes)
lambd = 3e-3                                             # Weight decay parameter 
hidden_patch_size =14
hidden_layer_size = hidden_patch_size*hidden_patch_size  # hidden layer Size
rho = 0.1                                                # desired average activation of the hidden units.                        
beta = 3                                                 # weight of sparsity penalty term   

### load data
# Load the dataset
f = gzip.open('/Users/wanyanxie/Summer_2014_Study/Deep_Learning/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

######## Simulate a Labeled and Unlabeled set
#labeled
labeledSet   = np.where((train_set[1] > 0) & (train_set[1] < 4))[0]
numTrain = round(len(labeledSet)/2)
trainSet = labeledSet[1:numTrain]
testSet  = labeledSet[numTrain+1:]
trainData   = train_set[0][trainSet, :]
trainLabels = train_set[1][trainSet] 
testData = train_set[0][testSet, :]
testLabels = train_set[1][testSet] 

#unlabeled 
unlabeledSet = np.where(train_set[1] >= 5)[0]
unlabeledData = train_set[0][unlabeledSet, :]


############# train
iterations = 200
model=autoencoder(input_layer_size, hidden_layer_size, beta, rho, lambd)
input_data = unlabeledData.T
theta= scipy.optimize.minimize(model.autoencoder_Cost_Grad, x0=model.theta, 
                                 args = (input_data,), 
                                 method = 'L-BFGS-B', 
                                 jac = True, 
                                 options = {'maxiter': iterations})  
                                  

W1 = theta.x[0:model.W1_dim].reshape(hidden_layer_size,input_layer_size)
b1 = theta.x[model.W1_dim+model.W2_dim: model.W1_dim+model.W2_dim +model.b1_dim].reshape(hidden_layer_size,1) 

showHiddenIMAGES(W1,patch_size,hidden_patch_size)

############# Extracting features
feedForward_train = feedForwardAutoencoder(W1,b1, trainData.T)
feedForward_test = feedForwardAutoencoder(W1,b1, testData.T)
a2_train = feedForward_train.hidden_layer_activiation()
a2_test = feedForward_test.hidden_layer_activiation()


############# Training and testing the logistic regression model
iterations = 100
option = {'maxiter': iterations}
theta_labeled = softmaxTrain (hidden_layer_size, numLabels, lambd, (a2_train.T,trainLabels ), option)
pred = softmaxPredict(theta_labeled, (a2_test.T,testLabels)) 