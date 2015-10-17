from matplotlib import pyplot
def DisplayIMAGES(W,nrow, ncol, patch_size):
    fig,axis = pyplot.subplots(nrows=nrow,ncols=ncol )
    i=0   
    for axis in axis.flat:
        axis.imshow(W[i].reshape(patch_size, patch_size),cmap='gray',interpolation = 'nearest')
        i+=1
        axis.set_frame_on(False)
        axis.set_axis_off()  
        pyplot.subplots_adjust(hspace=0,wspace=0)     
    pyplot.show()