

import sklearn.naive_bayes
import numpy as np
from dataset import fer2013
import numpy as np
import scipy
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

(train_x, train_y), (val_x,val_y), (test_x, test_y) = fer2013(seed=1214995756)
train_x = train_x[0:10000,:]
train_y = train_y[0:10000]
#(28709, 2304)

"""
dimensionality reduction code

m = train_x.shape[0]
n = train_x.shape[1]

Mu = np.mean(train_x, axis=0)
print(Mu.shape)
train_x = train_x - Mu

Sigma = (train_x.T).dot(train_x)/(m-1)
U,S,V = scipy.linalg.svd(Sigma)

V = V[:,0:k]
T = train_x.dot(V)

"""

GNB = sklearn.naive_bayes.GaussianNB()
GNB.fit(train_x,train_y)
y_pred = GNB.predict(val_x)
lin_accuracy=np.mean(y_pred == val_y)
print("GNB accuracy = " + str(lin_accuracy))


#plt.show()
#plt.savefig('/results/loss_plot.png')