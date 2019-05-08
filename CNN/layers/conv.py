import numpy as np
import scipy.signal


class ConvLayer(object):
    def __init__(self, n_i, n_o, h):
        # glorot initialization

        self.n_i = n_i
        self.n_o = n_o

        f_o = n_o*h*h
        f_i = n_i*h*h

        self.W = np.random.normal(0,(np.sqrt(2)/np.sqrt(f_o+f_i)), (n_o,n_i,h,h))
        self.b = np.zeros((1,n_o)).astype('float64')

        self.W_grad = None
        self.b_grad = None

    def forward(self, x):
        self.x = x.copy()
        #padding should depend on filter size
        one_sided_pad = (self.W.shape[2] - 1) / 2
        x_padded = np.pad(x,((0,0), (0,0), (one_sided_pad, one_sided_pad), (one_sided_pad, one_sided_pad)),mode='constant',constant_values=(0))
        output = np.zeros([self.x.shape[0],self.W.shape[0],self.x.shape[2],self.x.shape[3]]) #nb x no x nr x nc
        for j in range(self.x.shape[0]): #batch iterator
            for i in range(self.W.shape[0]): #output channels iterator
                output[j, i, :, :] = \
                    np.flip(np.flip(scipy.signal.correlate(self.W[i:i + 1, :, :, :], x_padded[j:j+1,:,:,:], mode='valid'), 2), 3) + self.b[0,i]

        return output

    def backward(self, y_grad):
        #Gradient of loss with respect to b
        self.b_grad = np.sum(y_grad,axis=(0,2,3)) #sum over all except output channels, 1 x no

        #Gradient of loss with respect to input to layer
        dl_dx = np.zeros(self.x.shape) #nb x ni x nr x nc
        #padding y_grad
        one_sided_pad = (self.W.shape[2] - 1) / 2
        y_grad_padded = np.pad(y_grad, ((0, 0), (0, 0), (one_sided_pad, one_sided_pad), (one_sided_pad, one_sided_pad)),
                          mode='constant', constant_values=(0))
        for sample in range(self.x.shape[0]): #batch iterator
            for o in range(self.W.shape[0]):  # for each channel output
                for i in range(self.x.shape[1]): #for each input channel
                    dl_dx[sample:sample + 1, i:i+1, :, :] = dl_dx[sample:sample + 1, i:i+1, :, :] + \
                                                        scipy.signal.convolve(self.W[o:o + 1, i:i+1, :, :],
                                                                              y_grad_padded[sample:sample + 1, o:o + 1,
                                                                              :, :], mode='valid')



        #Gradient of loss with respect to W
        one_sided_pad = (self.W.shape[2] - 1) / 2
        x_padded = np.pad(self.x, ((0, 0), (0, 0), (one_sided_pad, one_sided_pad), (one_sided_pad, one_sided_pad)),
                          mode='constant', constant_values=(0))
        dl_dW = np.zeros(self.W.shape)
        for o in range (self.W.shape[0]):#for each channel output
            for i in range(self.x.shape[1]): #for each input channel
                dl_dW[o:o+1,i:i+1,:,:] = np.flip(np.flip(scipy.signal.correlate(y_grad[:,o:o+1,:,:],x_padded[:,i:i+1,:,:],mode='valid'), 2), 3)
        self.W_grad = dl_dW

        return dl_dx

    def update_param(self, lr):
        self.W = self.W - (lr * self.W_grad)
        self.b = self.b - (lr * self.b_grad)
