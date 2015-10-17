import numpy as np
from matplotlib import pyplot

def Whitening(all_patches, epsilon):
    patches= all_patches-np.mean(all_patches, axis=0) 
    sigma = np.dot(patches,np.transpose(patches))/np.shape(patches)[1]
    U, s, V = np.linalg.svd(sigma,full_matrices=True)
        
    """Rotating the Data"""
    patches_rot= np.dot(np.transpose(U),patches)
    
    """ whitening"""
    patches=np.dot(U,(np.multiply(patches_rot.T, 1/np.sqrt(s+epsilon))).T)
    return patches
              
#def check_covariance (patches_rot):
#        
#        """ check covariance matrix of x_rot """
#        num_patches = np.shape(patches_rot)[1]
#        pyplot.imshow(np.dot(patches_rot,np.transpose(patches_rot)/num_patches),interpolation = 'nearest') 
#        pyplot.gca().invert_yaxis()
#        pyplot.colorbar()
#        pyplot.show()
    