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
from stackedAECost import stackedAE
from checkNumGrad_FineTuning import checkNumgrad
from feedForwardstackedAEC import feedForwardstackedAEC


###  Initialize constants and parameters 
patch_size=28
input_layer_size = patch_size * patch_size               # Size of input vector (MNIST images are 28x28)
numLabels  = 10                                          # Number of classes (MNIST images fall into 10 classes)
lambd = 3e-3                                             # Weight decay parameter 
hidden_layer_size_1 = 200 # 200                               # hidden layer 1 Size
hidden_layer_size_2 = 200  #200                              # hidden layer 2 Size

rho = 0.1                                                # desired average activation of the hidden units.                        
beta = 3                                                 # weight of sparsity penalty term   

####### load data
f = gzip.open('/Users/wanyanxie/Summer_2014_Study/Deep_Learning/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

####debug
#train_set= (train_set[0][1:50],train_set[1][1:50])
#######  Train the first sparse autoencoder
iterations = 200
model_1=autoencoder(input_layer_size, hidden_layer_size_1, beta, rho, lambd)
theta_1= scipy.optimize.minimize(model_1.autoencoder_Cost_Grad, x0=model_1.theta, 
                                 args = (train_set[0].T,), 
                                 method = 'L-BFGS-B', 
                                 jac = True, 
                                 options = {'maxiter': iterations}) 
                                  
W1_1 = theta_1.x[0:model_1.W1_dim].reshape(hidden_layer_size_1,input_layer_size)
b1_1 = theta_1.x[model_1.W1_dim+model_1.W2_dim: model_1.W1_dim+model_1.W2_dim +model_1.b1_dim].reshape(hidden_layer_size_1,1) 

#######  Train the second sparse autoencoder
iterations = 200
feedForward_train_1 = feedForwardAutoencoder(W1_1,b1_1, train_set[0].T)
a2_train = feedForward_train_1.hidden_layer_activiation()

model_2=autoencoder(hidden_layer_size_1, hidden_layer_size_2, beta, rho, lambd)
theta_2= scipy.optimize.minimize(model_2.autoencoder_Cost_Grad, x0=model_2.theta, 
                                 args = (a2_train,), 
                                 method = 'L-BFGS-B', 
                                 jac = True, 
                                 options = {'maxiter': iterations})  
W1_2 = theta_2.x[0:model_2.W1_dim].reshape(hidden_layer_size_2,hidden_layer_size_1)
b1_2 = theta_2.x[model_2.W1_dim+model_2.W2_dim: model_2.W1_dim+model_2.W2_dim +model_2.b1_dim].reshape(hidden_layer_size_2,1) 
                                 

                                                              
####### Train the softmax classifier on the L2 features
iterations = 200
option = {'maxiter':iterations}
feedForward_train_2 = feedForwardAutoencoder(W1_2,b1_2, a2_train)
a2_train_2 = feedForward_train_2.hidden_layer_activiation()
W1_3 = softmaxTrain (hidden_layer_size_2, numLabels, lambd, (a2_train_2.T,train_set[1] ), option)




##### prediction without fine tuning 
pre_theta= np.concatenate ((W1_1.flatten(), W1_2.flatten(),W1_3.flatten(), b1_1.flatten(), b1_2.flatten())) 
feedForward_val_woFN=feedForwardstackedAEC(input_layer_size, hidden_layer_size_1,
                                      hidden_layer_size_2, numLabels,pre_theta, 
                                      valid_set[0].T)
a2_val_woFN = feedForward_val_woFN.hidden_layer_activiation()
pred_woFN = a2_val_woFN.argmax(axis=0)
accuracy_woFN = float(np.sum(np.equal(pred_woFN,  valid_set[1])))/len(pred)

######## fine_tuning

#import stackedAECost
#reload(stackedAECost)
#from stackedAECost import stackedAE
#from checkNumGrad_FineTuning import checkNumgrad

fine_model = stackedAE(input_layer_size, hidden_layer_size_1, hidden_layer_size_2, numLabels, lambd,pre_theta)
#fine_model.stackedAE_Cost_Grad(fine_model.theta,train_set)
#### check gradient
#fine_model_debug = stackedAE(input_layer_size, 5, 5, numLabels, lambd,pre_theta)
#data_set_debug = (train_set[0][1:50],train_set[1][1:50])
#check=checkNumgrad (epsilon,data_set_debug,fine_model)  


#### fine tuning
#train_set_debug = (train_set[0][1:5000],train_set[1][1:5000])
iterations=200
theta_FN= scipy.optimize.minimize(fine_model.stackedAE_Cost_Grad, x0=fine_model.theta, 
                                 args = (train_set,), 
                                 method = 'L-BFGS-B', 
                                 jac = True, 
                                 options = {'maxiter': iterations})  



######## prediction with FN
#import feedForwardstackedAEC
#reload(feedForwardstackedAEC)
#from feedForwardstackedAEC import feedForwardstackedAEC
feedForward_val=feedForwardstackedAEC(input_layer_size, hidden_layer_size_1,
                                      hidden_layer_size_2, numLabels,theta_FN.x, 
                                      valid_set[0].T)
a2_val = feedForward_val.hidden_layer_activiation()
pred = a2_val.argmax(axis=0)
accuracy = float(np.sum(np.equal(pred,  valid_set[1])))/len(pred)
print accuracy
#feedForward_val_1 = feedForwardAutoencoder(W1_1,b1_1, test_set[0].T)
#a2_val_1 = feedForward_val_1.hidden_layer_activiation()
#feedForward_val_2 = feedForwardAutoencoder(W1_2,b1_2, a2_val_1)
#a2_val_2 = feedForward_val_2.hidden_layer_activiation()
#pred_val = softmaxPredict(theta_labeled, (a2_val_2.T,valid_set[1])) 

                              
                                                            
                                                                                                                        