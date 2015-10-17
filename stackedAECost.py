import numpy as np
from scipy.sparse import *
###########################################################################################
""" sparse autoencoder class """

class stackedAE(object):
        
   def __init__ (self,input_layer_size, hidden_layer_size_1, hidden_layer_size_2, numLabels, lambd, theta):
      self.input_layer_size = input_layer_size            # number of input layer units
      self.hidden_layer_size_1 = hidden_layer_size_1      # number of hidden layer units     
      self.hidden_layer_size_2 = hidden_layer_size_2      # number of hidden layer units     
      self.numLabels = numLabels      
      self.lambd = lambd                                  # weight decay parameter
      self.W1_dim= hidden_layer_size_1*input_layer_size   # W_ij: connection between unit j in layer l, and unit i in layer l+1.
      self.W2_dim= hidden_layer_size_2*hidden_layer_size_1
      self.W3_dim= numLabels*hidden_layer_size_2
      self.b1_dim= hidden_layer_size_1                    # b_i:bias term associated with unit i in layer textstyle l+1.                   
      self.b2_dim= hidden_layer_size_2
      self.theta = theta
      
 #     """ Initialization of Autoencoder object, W1->[-lim,lim], b1,b2-> 0 """   
 #     lim_1 = np.sqrt(6.0/(input_layer_size + hidden_layer_size_1 +1)) 
 #     lim_2 =  np.sqrt(6.0/(hidden_layer_size_1 + hidden_layer_size_2 +1)) 
 #     W1 =  np.array(np.random.uniform(-lim_1,lim_1,size=(hidden_layer_size_1,input_layer_size)))
 #     W2 =  np.array(np.random.uniform(-lim_2,lim_2,size=(hidden_layer_size_2,hidden_layer_size_1)))
 #     W3 =  0.005*np.random.normal(0,1, self.numLabels *self.hidden_layer_size_2)
 #     b1 =  np.array(np.zeros(hidden_layer_size_1))
 #     b2 =  np.array(np.zeros(hidden_layer_size_2)) 
 #     b3 =  np.array(np.zeros(numLabels)) 
 #     """ unroll W1, W2, b1, b2 to theta """
 #     #self.theta= np.concatenate ((W1.flatten(), W2.flatten(),W3, b1.flatten(), b2.flatten())) 
 #     self.theta= np.concatenate ((W1.flatten(), W2.flatten(),W3, b1.flatten(), b2.flatten(),b3.flatten())) 
 
       
   def sigmoid (self,z):     
       return 1/(1+np.exp(-z)) 
   
   def stackedAE_Cost_Grad(self,theta,data_set):
       labels = data_set[1]
       input_data = data_set[0].T
       sample_size= np.shape(input_data)[1]
       
       """ get weights and biases from theta """ 
       dim1=self.W1_dim
       dim2=self.W1_dim+self.W2_dim
       dim3=self.W1_dim+self.W2_dim + self.W3_dim
       dim4 =self.W1_dim+self.W2_dim + self.W3_dim +self.b1_dim
       dim5 =self.W1_dim+self.W2_dim + self.W3_dim +self.b1_dim + self.b2_dim
       
       W1 =  theta[0:dim1].reshape(self.hidden_layer_size_1,self.input_layer_size)
       W2 =  theta[dim1: dim2 ].reshape(self.hidden_layer_size_2, self.hidden_layer_size_1) 
       W3 =  theta[dim2: dim3 ].reshape(self.numLabels, self.hidden_layer_size_2) 
 
       b1 =  theta[dim3: dim4].reshape(self.hidden_layer_size_1,1) 
       b2 =  theta[dim4: dim5].reshape(self.hidden_layer_size_2,1) 
       #b3 =  theta[dim5:].reshape(self.numLabels,1) 

       """ foward propgation """ 
       z2 = np.dot(W1,input_data) + b1   # z_i total weighted sum of inputs to unit i in layer l
       a2 = self.sigmoid(z2)             # activation of z_i, a1 =inputs x
       z3 = np.dot(W2,a2) +  b2
       a3 = self.sigmoid(z3)              # hypothesis on inputsx
       ### output layer
       #z4 = np.dot(W3,a3) + b3
       z4 = np.dot(W3,a3)      
       z4 = z4- np.max(z4,0)
       h=np.multiply(np.exp(z4),  1.0/np.sum(np.exp(z4) ,0))     

       """ Cost function """  
       #sq_error = 0.5/sample_size* np.sum(np.multiply (input_data-h,input_data-h))  # J(W,b)
       #regularization= 0.5*self.lambd*(np.sum(np.multiply(W1,W1))+np.sum(np.multiply(W2,W2))) # weight decay term
       #J= sq_error + regularization 
       groundTruth = csc_matrix( (np.repeat(1,sample_size),(range(sample_size),labels)), shape=(sample_size,self.numLabels)).todense()
          
       J = -1.0/sample_size*np.sum(np.multiply(groundTruth,np.log(h).T)) + 0.5*self.lambd*np.sum(np.multiply (W3,W3)) 

       
       """ backword propagation : calculate dJ_dz """ 
       ### output_layer
       delta4 =  -np.array(groundTruth-h.T).T
       delta3 =  np.multiply(np.dot(W3.T, delta4), np.multiply(a3, 1-a3))   
       delta2 =  np.multiply(np.dot(W2.T, delta3), np.multiply(a2, 1-a2)) 
       
       """ backword propagation : calculate dJ_dW, dJ_db  """
      
       #dJ_dW1 = np.array(np.dot(delta2, np.transpose(input_data)) /  sample_size+self.lambd * W1)
       #dJ_dW2 = np.array(np.dot(delta3, np.transpose(a2)) / sample_size + self.lambd * W2)
       dJ_dW1 = np.array(np.dot(delta2, np.transpose(input_data)) /  sample_size)
       dJ_dW2 = np.array(np.dot(delta3, np.transpose(a2)) / sample_size)
       dJ_dW3 = np.array(np.dot(delta4, np.transpose(a3)) / sample_size + self.lambd * W3)
       dJ_db1 = np.array(np.sum(delta2,axis=1) /  sample_size)
       dJ_db2 = np.array(np.sum(delta3,axis=1).reshape(self.hidden_layer_size_2,1) / sample_size)
       #dJ_db3 = np.array(np.sum(delta4,axis=1).reshape(self.numLabels,1) / sample_size)
       """ unroll dJ_dW, dJ_db to d_theta """
       #d_theta= np.concatenate ((dJ_dW1.flatten(), dJ_dW2.flatten(),dJ_dW3.flatten(),dJ_db1.flatten(), dJ_db2.flatten(),dJ_db3.flatten()))
       d_theta= np.concatenate ((dJ_dW1.flatten(), dJ_dW2.flatten(),dJ_dW3.flatten(),dJ_db1.flatten(), dJ_db2.flatten()))

       return [J, d_theta]


###### try

#W1_dim= hidden_layer_size_1*input_layer_size    
#W2_dim= hidden_layer_size_2*hidden_layer_size_1
#W3_dim= numLabels*hidden_layer_size_2
#b1_dim= hidden_layer_size_1                             
#b2_dim= hidden_layer_size_2
#
#lim_1 = np.sqrt(6.0/(input_layer_size + hidden_layer_size_1 +1)) 
#lim_2 =  np.sqrt(6.0/(hidden_layer_size_1 + hidden_layer_size_2 +1)) 
#W1 =  np.array(np.random.uniform(-lim_1,lim_1,size=(hidden_layer_size_1,input_layer_size)))
#W2 =  np.array(np.random.uniform(-lim_2,lim_2,size=(hidden_layer_size_2,hidden_layer_size_1)))
#W3 =  0.005*np.random.normal(0,1, numLabels *hidden_layer_size_2)
#b1 =  np.array(np.zeros(hidden_layer_size_1))
#b2 =  np.array(np.zeros(hidden_layer_size_2)) 
#b3 =  np.array(np.zeros(numLabels)) 
#theta_try= np.concatenate ((W1.flatten(), W2.flatten(),W3, b1.flatten(), b2.flatten(),b3.flatten()))
#dim1=W1_dim
#dim2=W1_dim+W2_dim
#dim3=W1_dim+W2_dim + W3_dim
#dim4 =W1_dim+W2_dim + W3_dim +b1_dim
#dim5=W1_dim+W2_dim + W3_dim +b1_dim + b2_dim
###
#theta_new = fine_model.theta + epsilon_vector
#W1 = fine_model.theta[0:dim1].reshape(hidden_layer_size_1,input_layer_size)
#W2 = fine_model.theta[dim1: dim2 ].reshape(hidden_layer_size_2, hidden_layer_size_1) 
#W3 = fine_model.theta[dim2: dim3 ].reshape(numLabels, hidden_layer_size_2) 
#b1 =  fine_model.theta[dim3: dim4].reshape(hidden_layer_size_1,1) 
#b2 =  fine_model.theta[dim4:dim5].reshape(hidden_layer_size_2,1) 
#
#W1_new = theta_new[0:dim1].reshape(hidden_layer_size_1,input_layer_size)
#W2_new = theta_new[dim1: dim2 ].reshape(hidden_layer_size_2, hidden_layer_size_1) 
#W3_new = theta_new[dim2: dim3 ].reshape(numLabels, hidden_layer_size_2) 
#b1_new = theta_new[dim3: dim4].reshape(hidden_layer_size_1,1) 
#b2_new = theta_new[dim4:dim5].reshape(hidden_layer_size_2,1)
#
#input_data = train_set[0].T
#
#z2 = np.dot(W1,input_data) + b1   
#a2 = fine_model.sigmoid(z2)             
#z3 = np.dot(W2,a2) +  b2
#a3 = fine_model.sigmoid(z3)             
#z4 = np.dot(W3,a3)      
#z4 = z4- np.max(z4,0)
#h=np.multiply(np.exp(z4),  1.0/np.sum(np.exp(z4) ,0))
#
#
#z2_new = np.dot(W1_new,input_data) + b1_new   
#a2_new = fine_model.sigmoid(z2_new)             
#z3_new = np.dot(W2_new,a2_new) +  b2_new
#a3_new = fine_model.sigmoid(z3_new)             
#z4_new = np.dot(W3_new,a3_new)      
#z4_new = z4_new- np.max(z4_new,0)
#h_new=np.multiply(np.exp(z4_new),  1.0/np.sum(np.exp(z4_new) ,0))
#
#sample_size= np.shape(input_data)[1]
#groundTruth = csc_matrix( (np.repeat(1,sample_size),(range(sample_size),train_set[1])), shape=(sample_size,numLabels)).todense()   
#
#
#J = -1.0/sample_size*np.sum(np.multiply(groundTruth,np.log(h).T)) + 0.5*fine_model.lambd*np.sum(np.multiply (W3,W3)) 
#J_new = -1.0/sample_size*np.sum(np.multiply(groundTruth,np.log(h_new).T)) + 0.5*fine_model.lambd*np.sum(np.multiply (W3_new,W3_new)) 

#delta4 =  np.array(groundTruth-h.T).T
#delta3 =  np.multiply(np.dot(W3.T, delta4), np.multiply(a3, 1-a3))   
#delta2 =  np.multiply(np.dot(W2.T, delta3), np.multiply(a2, 1-a2))
#                               
#dJ_dW1 = np.array(np.dot(delta2, np.transpose(input_data)) /  sample_size+lambd * W1)
#dJ_dW2 = np.array(np.dot(delta3, np.transpose(a2)) / sample_size + lambd * W2)
#dJ_dW3 = np.array(np.dot(delta4, np.transpose(a3)) / sample_size + lambd * W3)
#dJ_db1 = np.array(np.sum(delta2,axis=1) /  sample_size)
#dJ_db2 = np.array(np.sum(delta3,axis=1).reshape(hidden_layer_size_2,1) / sample_size) 
#dJ_db3 = np.array(np.sum(delta4,axis=1).reshape(numLabels,1) / sample_size)                                 
#d_theta= np.concatenate ((dJ_dW1.flatten(), dJ_dW2.flatten(),dJ_dW3.flatten(),dJ_db1.flatten(), dJ_db2.flatten()))
#J = -1.0/sample_size*np.sum(np.multiply(groundTruth,np.log(h).T)) + 0.5*lambd*np.sum(np.multiply (W3,W3))
##
