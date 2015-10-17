from matplotlib import pyplot
def showHiddenIMAGES(W,patch_size,hidden_patch_size):
    fig,axis = pyplot.subplots(nrows=hidden_patch_size,ncols=hidden_patch_size )
    i=0   
    for axis in axis.flat:
        axis.imshow(W[i].reshape(patch_size, patch_size),cmap='gray',interpolation = 'nearest')
        i+=1
        axis.set_frame_on(False)
        axis.set_axis_off()  
        pyplot.subplots_adjust(hspace=0,wspace=0)     
    pyplot.show()