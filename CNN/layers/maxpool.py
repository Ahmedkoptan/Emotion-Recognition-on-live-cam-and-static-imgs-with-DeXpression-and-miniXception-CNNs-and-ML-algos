import numpy as np


class MaxPoolLayer(object):
    def __init__(self, size=2):
        self.locs = None  # to store max locations
        self.size = size  # size of the pooling

    def forward(self, x):
        row_iters = int(x.shape[2]//self.size)
        col_iters = int(x.shape[3]//self.size)
        output = np.zeros([x.shape[0],x.shape[1],row_iters,col_iters])
        self.locs = np.zeros(x.shape)
        for sample in range(x.shape[0]):
            for channel in range(x.shape[1]):
                for row in range(row_iters):
                    for col in range(col_iters):
                        output[sample:sample + 1, channel:channel + 1, row:row + 1, col:col + 1] = np.amax(
                            x[sample:sample + 1, channel:channel + 1,
                            row * self.size : (row + 1) * self.size ,
                            col * self.size : (col + 1) * self.size ], axis=None)
                        k = np.argwhere(x[sample:sample + 1, channel:channel + 1,
                            row * self.size : (row + 1) * self.size ,
                            col * self.size : (col + 1) * self.size ] == output[sample:sample + 1, channel:channel + 1, row:row + 1, col:col + 1])
                        self.locs[sample,channel,k[:,2]+row * self.size,k[:,3]+col * self.size] = 1
        return output

    def backward(self, y_grad):
        dl_dx = np.zeros(self.locs.shape)
        for sample in range(y_grad.shape[0]):
            for channel in range(y_grad.shape[1]):
                for row in range(y_grad.shape[2]):
                    for col in range(y_grad.shape[3]):
                        u = np.argwhere(self.locs[sample:sample + 1, channel:channel + 1,
                        row * self.size: (row + 1) * self.size,
                        col * self.size: (col + 1) * self.size] == 1)
                        #print u
                        dl_dx[sample:sample+1,channel:channel+1,u[:,2]+row * self.size,u[:,3]+col * self.size] = \
                            y_grad[sample:sample+1,channel:channel+1,row:row+1,col:col+1]

        return dl_dx

    def update_param(self, lr):
        pass
