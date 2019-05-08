class ReluLayer(object):
    def __init__(self):
        """
        Rectified Linear Unit
        """
        self.y = None

    def forward(self, x):

        Yu = x.copy()
        Yu[Yu<=0] = 0.0
        self.y = Yu
        return Yu

    def backward(self, y_grad):
        u = self.y.copy()
        u[u>0] = 1.0
        return (y_grad)*u

    def update_param(self, lr):
        pass  # no parameters to update
