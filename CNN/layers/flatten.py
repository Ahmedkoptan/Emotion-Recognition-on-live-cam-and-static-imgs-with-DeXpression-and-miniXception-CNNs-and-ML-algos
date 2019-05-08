import numpy as np

class FlattenLayer(object):
    def __init__(self):
        """
        Flatten layer
        """
        self.orig_shape = None # to store the shape for backpropagation

    def forward(self, x):
        self.orig_shape = x.shape
        return np.reshape(x.copy(),[-1,x.shape[1]*x.shape[2]*x.shape[3]])

    def backward(self, y_grad):
        return np.reshape(y_grad.copy(),self.orig_shape)

    def update_param(self, lr):
        pass
