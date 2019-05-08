import numpy as np


class SoftMaxLayer(object):
    def __init__(self):
        """
        Constructor
        """
        self.y = None

    def forward(self, x):
        X=x.copy()
        X = X-np.amax(X)
        u = [np.sum(np.exp(X.copy()),axis=1)]
        self.y = ((np.exp(X).T)/u).T
        return ((np.exp(X).T)/u).T

    def backward(self, y_grad): ####
        temp=np.zeros((len(self.y[0,:]),len(self.y[0,:])))
        u=np.zeros((len(self.y[:,0]),len(self.y[0,:])))
        for i in range(len(self.y[:,0])):
            temp = np.diag(self.y[i,:])-((self.y[i:i+1,:].T).dot(self.y[i:i+1,:]))
            u[i,:] = y_grad[i:i+1,:].dot(temp)
        return u

    def update_param(self, lr):
        pass  # no learning for softmax layer
