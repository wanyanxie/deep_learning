import numpy as np
def softmaxPredict(theta, valid_set):
    val_x = valid_set[0]                                                                                    
    M = np.dot(theta,val_x.T)
    new_M = M - np.max(M,0)
    exp_M = np.exp(new_M) 
    h= np.multiply(exp_M,  1.0/np.sum(exp_M,0))
    pred = h.argmax(axis=0)
    accuracy = float(np.sum(np.equal(pred,  valid_set[1])))/len(pred)  
    return (pred,accuracy)