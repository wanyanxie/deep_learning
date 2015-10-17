import numpy as np
class feedForwardstackedAEC(object):
      def __init__ (self,input_layer_size, hidden_layer_size_1, hidden_layer_size_2, numLabels,theta, data):
          self.theta = theta           
          self.data=data
          self.input_layer_size = input_layer_size            # number of input layer units
          self.hidden_layer_size_1 = hidden_layer_size_1      # number of hidden layer units     
          self.hidden_layer_size_2 = hidden_layer_size_2      # number of hidden layer units     
          self.numLabels = numLabels      
          self.W1_dim= hidden_layer_size_1*input_layer_size   # W_ij: connection between unit j in layer l, and unit i in layer l+1.
          self.W2_dim= hidden_layer_size_2*hidden_layer_size_1
          self.W3_dim= numLabels*hidden_layer_size_2
          self.b1_dim= hidden_layer_size_1                    # b_i:bias term associated with unit i in layer textstyle l+1.                   
          self.b2_dim= hidden_layer_size_2
      
          
      def sigmoid (self,z):     
          return 1/(1+np.exp(-z)) 
   
      def hidden_layer_activiation(self):
          dim1=self.W1_dim
          dim2=self.W1_dim+self.W2_dim
          dim3=self.W1_dim+self.W2_dim + self.W3_dim
          dim4 =self.W1_dim+self.W2_dim + self.W3_dim +self.b1_dim
          dim5 =self.W1_dim+self.W2_dim + self.W3_dim +self.b1_dim + self.b2_dim
       
          self.W1 =  self.theta[0:dim1].reshape(self.hidden_layer_size_1,self.input_layer_size)
          self.W2 =  self.theta[dim1: dim2 ].reshape(self.hidden_layer_size_2, self.hidden_layer_size_1) 
          self.W3 =  self.theta[dim2: dim3 ].reshape(self.numLabels, self.hidden_layer_size_2) 
 
          self.b1 =  self.theta[dim3: dim4].reshape(self.hidden_layer_size_1,1) 
          self.b2 =  self.theta[dim4: dim5].reshape(self.hidden_layer_size_2,1) 
          
          """ foward propgation """ 
          z2 = np.dot(self.W1,self.data) + self.b1   # z_i total weighted sum of inputs to unit i in layer l
          a2 = self.sigmoid(z2)             # activation of z_i, a1 =inputs x
          z3 = np.dot(self.W2,a2) +  self.b2
          a3 = self.sigmoid(z3)              # hypothesis on inputsx
          z4 = np.dot(self.W3,a3)      
          z4 = z4- np.max(z4,0)
          h=np.multiply(np.exp(z4),  1.0/np.sum(np.exp(z4) ,0))                  
          return h
