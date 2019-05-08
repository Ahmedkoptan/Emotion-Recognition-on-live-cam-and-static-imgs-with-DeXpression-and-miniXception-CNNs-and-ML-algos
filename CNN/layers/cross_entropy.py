import numpy as np


class CrossEntropyLayer(object):
    def __init__(self):
        """
        Constructor
        """
        self.x = None
        self.t = None

    def forward(self, x, t):
        self.x = x.copy()
        self.t = t.copy()
        return np.sum(np.log(x)*t)/-len(x[:,0])

    def backward(self, y_grad=None):
        return -(self.t/(self.x))/len(self.x[:,0])
