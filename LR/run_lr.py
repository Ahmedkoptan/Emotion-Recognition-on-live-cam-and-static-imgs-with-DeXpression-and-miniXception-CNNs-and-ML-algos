# For FER2013

import sklearn.linear_model
import numpy as np
from FER2013.LR.dataset import fer2013
import numpy as np
import scipy
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from FER2013.LR.PCA import PCA_dim_red

(train_x, train_y), (val_x,val_y), (test_x, test_y) = fer2013(seed=1214995756)

#print(train_x, train_y)

#train_x, val_x, test_x = PCA_dim_red(train_x, val_x, test_x, 0.99)



# Using C as regularization parameter
C_range = [0.001,0.01,0.1,1,10,100]

for i in C_range:
    LR = sklearn.linear_model.LogisticRegression(penalty='l2', C=i, solver='lbfgs', multi_class='multinomial',max_iter=10000)
    LR.fit(train_x, train_y)
    y_pred = LR.predict(test_x)
    acc = sklearn.metrics.accuracy_score(y_pred, test_y)
    print("LR accuracy = " + str(acc) + "  for C=" + str(i))


#plt.show()
#plt.savefig('/results/loss_plot.png')
