import numpy as np


class FullLayer(object):
    def __init__(self, n_i, n_o):
        self.x = None
        self.W_grad = None
        self.b_grad = None

        self.W = np.random.normal(0, (np.sqrt(2)/np.sqrt(n_o+n_i)), (n_o,n_i))
        self.b = np.zeros((1,n_o)).astype('float64')

    def forward(self, x):
        self.x = x.copy()
        return (x.dot(self.W.T))+self.b

    def backward(self, y_grad):
        k = np.ones(y_grad.shape)
        self.b_grad = ((y_grad.T).dot(k[:,1:2])).T
        self.W_grad = (y_grad.T).dot(self.x)
        return y_grad.dot(self.W)

    def update_param(self, lr):
        self.W = self.W - (lr*self.W_grad)
        self.b = self.b - (lr*self.b_grad)
