import cPickle, gzip
import numpy as np
from scipy.sparse import *
import scipy.optimize

###  Initialize constants and parameters 
inputSize = 28 * 28; # Size of input vector (MNIST images are 28x28)
numClasses = 10;     # Number of classes (MNIST images fall into 10 classes)
lambd = 1e-4;        # Weight decay parameter
epsilon=1e-4         # tolerance

# Load the dataset
f = gzip.open('/Users/wanyanxie/Summer_2014_Study/Deep_Learning/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

###########################################################################################
""" sparse autoencoder class """
class softmax(object):
      def __init__ (self,numClasses, inputSize, lambd):
          self.numClasses = numClasses               # number of classes
          self.inputSize = inputSize                 # number input samples    
          self.lambd = lambd                         # number of output layer units
          self.theta = 0.005*np.random.normal(0,1, self.numClasses *self.inputSize)
          
      def softmax_Cost_Grad(self,theta,data_set):   
          theta = theta.reshape(self.numClasses, self.inputSize)
          labels = data_set[1]
          train_x = data_set[0]
          #debug
          #labels = data_set[1][1:500]
          #train_x = data_set[0][1:500]
          numCases = np.shape(train_x)[0]
          groundTruth = csc_matrix( (np.repeat(1,numCases),(range(numCases),labels)), shape=(numCases,self.numClasses)).todense()
              
          M = np.dot(theta,train_x.T)
          new_M = M - np.max(M,0)
          exp_M = np.exp(new_M) 
          h=np.multiply(exp_M,  1.0/np.sum(exp_M,0))
          
          J = -1.0/numCases*np.sum(np.multiply(groundTruth,np.log(h).T)) + 0.5*self.lambd*np.sum(np.multiply (theta,theta)) 
          thetagrad = -1.0/numCases*np.array(np.dot((groundTruth - h.T).T,train_x)) + self.lambd *theta 
          thetagrad = thetagrad.flatten()
          return [J, thetagrad]

         
def checkNumgrad (epsilon,data_set, inputSize) :
    """ Gradient checking """
    model=softmax(numClasses, inputSize, lambd)
    theta=model.theta
    epsilon_vector = np.array(np.zeros(np.size(theta)))
    numGrad = np.array(np.zeros(np.size(theta)))
    for p in range (np.size(theta)):
        print p
        epsilon_vector[p] = epsilon
        J1_grad=model.softmax_Cost_Grad(theta+epsilon_vector,data_set)
        J2_grad=model.softmax_Cost_Grad(theta-epsilon_vector,data_set)
        numGrad[p] = (J1_grad[0]-J2_grad[0])/(2*epsilon)
        epsilon_vector[p] = 0    
    return numGrad, J1_grad[1] 
#check=checkNumgrad (epsilon,train_set,inputSize)  
#print max(check[0]/np.sum(check[0]*check[0])-check[1]/np.sum(check[1]*check[1]))           
###################################################### train
iterations = 100
option = {'maxiter': iterations}
data_set= train_set

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

theta =  softmaxTrain (inputSize, numClasses, lambd, train_set, option)  
    
def softmaxPredict(theta, valid_set):
    val_x = valid_set[0]                                                                                    
    M = np.dot(theta,val_x.T)
    new_M = M - np.max(M,0)
    exp_M = np.exp(new_M) 
    h= np.multiply(exp_M,  1.0/np.sum(exp_M,0))
    pred = h.argmax(axis=0)
    return pred
    
pred_val=softmaxPredict(theta, valid_set) 
accuracy = float(np.sum(np.equal(pred_val,  valid_set[1])))/len(pred_val)  
print "accuracy", accuracy