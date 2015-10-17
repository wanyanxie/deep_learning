import numpy as np

###########################################################################################
""" sparse autoencoder class """

class linear_autoencoder(object):
        
   def __init__ (self,input_layer_size, hidden_layer_size, beta, rho, lambd):
      self.input_layer_size = input_layer_size        # number of input layer units
      self.hidden_layer_size = hidden_layer_size      # number of hidden layer units     
      self.output_layer_size = input_layer_size       # number of output layer units
      self.beta = beta                                # weight of the sparsity penalty term  
      self.rho = rho                                  # sparsity parameter (desired level of sparsity) 
      self.lambd = lambd                              # weight decay parameter
      self.W1_dim= hidden_layer_size*input_layer_size # W_ij: connection between unit j in layer l, and unit i in layer l+1.
      self.W2_dim= self.output_layer_size*hidden_layer_size
      self.b1_dim= hidden_layer_size                  # b_i:bias term associated with unit i in layer textstyle l+1.                   
      self.b2_dim= self.output_layer_size
      
      """ Initialization of Autoencoder object, W1->[-lim,lim], b1,b2-> 0 """   
      lim= np.sqrt(6.0/(input_layer_size + self.output_layer_size +1)) 
      W1 =  np.array(np.random.uniform(-lim,lim,size=(hidden_layer_size,input_layer_size)))
      W2 =  np.array(np.random.uniform(-lim,lim,size=(self.output_layer_size,hidden_layer_size)))
      b1 =  np.array(np.zeros(hidden_layer_size))
      b2 =  np.array(np.zeros(self.output_layer_size)) 
      
      """ unroll W1, W2, b1, b2 to theta """
      self.theta= np.concatenate ((W1.flatten(), W2.flatten(), b1.flatten(), b2.flatten())) 
 
       
   def sigmoid (self,z):     
       return 1/(1+np.exp(-z)) 
   
   def linear_autoencoder_Cost_Grad(self,theta,input_data):
       sample_size= np.shape(input_data)[1]
       
       """ get weights and biases from theta """ 
       W1 =  theta[0:self.W1_dim].reshape(self.hidden_layer_size,self.input_layer_size)
       W2 =  theta[self.W1_dim: self.W1_dim+self.W2_dim ].reshape(self.output_layer_size, self.hidden_layer_size)  
       b1 =  theta[self.W1_dim+self.W2_dim: self.W1_dim+self.W2_dim +self.b1_dim].reshape(self.hidden_layer_size,1) 
       b2 =  theta[self.W1_dim+self.W2_dim + self.b1_dim:].reshape(self.output_layer_size,1) 
      
       """ foward propgation """ 
       z2 = np.dot(W1,input_data) + b1   # z_i total weighted sum of inputs to unit i in layer l
       a2 = self.sigmoid(z2)             # activation of z_i, a1 =inputs x
       z3 = np.dot(W2,a2) +  b2.reshape(self.output_layer_size,1)
      
       
       """ Sparsity term """ 
       rho_hat = np.matrix(np.sum(a2,axis=1)/sample_size)   # average activation of hidden unit i
       KL= np.sum(self.rho*np.log(self.rho/rho_hat) + (1-self.rho)* np.log((1-self.rho)/(1-rho_hat)))  # penalty term : penalizes rho_j deviating significantly from rho_hat
       dKL =self.beta * (-self.rho / rho_hat + (1 - self.rho) / (1 - rho_hat)).reshape(self.hidden_layer_size,1)
       
       """ Cost function """  
       sq_error = 0.5/sample_size* np.sum(np.multiply (input_data-z3,input_data-z3))  # J(W,b)
       regularization= 0.5*self.lambd*(np.sum(np.multiply(W1,W1))+np.sum(np.multiply(W2,W2))) # weight decay term
       J= sq_error + regularization + self.beta * KL
       
       """ backword propagation : calculate dJ_dz """ 
       delta3 = -(input_data-z3)  # y= inputs x
       delta2 =  np.multiply(np.dot(np.transpose(W2), delta3)+ dKL, np.multiply(a2, 1-a2))
       
       """ backword propagation : calculate dJ_dW, dJ_db  """
      
       dJ_dW1 = np.array(np.dot(delta2, np.transpose(input_data)) /  sample_size+self.lambd * W1)
       dJ_dW2 = np.array(np.dot(delta3, np.transpose(a2)) / sample_size + self.lambd * W2)
       dJ_db1 = np.array(np.sum(delta2,axis=1) /  sample_size)
       dJ_db2 = np.array(np.sum(delta3,axis=1).reshape(self.output_layer_size,1) / sample_size)
       
       """ unroll dJ_dW, dJ_db to d_theta """
       d_theta= np.concatenate ((dJ_dW1.flatten(), dJ_dW2.flatten(),dJ_db1.flatten(), dJ_db2.flatten()))
       return [J, d_theta]








                                   
                                  


