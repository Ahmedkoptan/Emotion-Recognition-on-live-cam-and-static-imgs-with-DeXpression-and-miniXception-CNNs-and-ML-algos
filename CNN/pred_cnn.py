from layers.conv import ConvLayer
from layers.flatten import FlattenLayer
from layers.maxpool import MaxPoolLayer
from layers.relu import ReluLayer
from layers.sequential import Sequential
from layers.softmax import SoftMaxLayer
from layers.full import FullLayer
from layers.cross_entropy import CrossEntropyLayer
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from layers.dataset import fer2013


#Import and process the input
(train_x, train_y), (val_x,val_y), (test_x, test_y) = fer2013()

lr = 0.1
epochs = 100
batch_size = 128
myNet = Sequential(layers=(ConvLayer(n_i=1,n_o=16,h=3),
                           ReluLayer(),
                           MaxPoolLayer(size=2),
                           ConvLayer(n_i=16,n_o=32,h=3),
                           ReluLayer(),
                           MaxPoolLayer(size=2),
                           FlattenLayer(),
                           FullLayer(n_i=12*12*32,n_o=7),
                           SoftMaxLayer()),
                   loss=CrossEntropyLayer())

myNet.load()
"""
pred = myNet.predict(val_x)
accuracy = np.mean(pred == val_y)
print('At learning rate = '+str(lr))
print('Validation Accuracy of Convolutional Neural Network = '+str(accuracy))
"""

forw = myNet.forward(test_x)
pred = myNet.predict(test_x)
accuracy = np.mean(pred == test_y)
print('At learning rate = '+str(lr))
print('Testing Accuracy of Convolutional Neural Network = '+str(accuracy))

ty = np.argmax(test_y,axis=1)
prediction = np.argmax(pred,axis=1)
###Confusion MAtrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ty, prediction)
print(cm)

"""
###Precision Recall
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

n_classes = 7
# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(test_y[:, i],
                                                        forw[:, i])
    average_precision[i] = average_precision_score(test_y[:, i], forw[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(test_y.ravel(),
    forw.ravel())
average_precision["micro"] = average_precision_score(test_y, forw,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))


from sklearn.utils.fixes import signature
# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.figure()
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
         where='post')
plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b',
                 **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    .format(average_precision["micro"]))
"""