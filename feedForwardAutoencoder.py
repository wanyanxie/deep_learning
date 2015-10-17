import numpy as np
class feedForwardAutoencoder(object):
      def __init__ (self,W1,b1, data):
          self.W1 = W1
          self.b1 = b1             
          self.data=data
          
      def sigmoid (self,z):     
          return 1/(1+np.exp(-z)) 
   
      def hidden_layer_activiation(self):
          
          """ foward propgation """ 
          z2 = np.dot(self.W1,self.data) + self.b1   # z_i total weighted sum of inputs to unit i in layer l
          a2 = self.sigmoid(z2)             # activation of z_i, a1 =inputs x
          return a2

                   