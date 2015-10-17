import numpy as np
from scipy.sparse import *

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