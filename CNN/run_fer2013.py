"""
At learning rate = 0.1, epochs = 15, batch size= 128, training only on the first 1500 images
Accuracy of Convolutional Neural Network = 0.8047207737929387

"""
from layers.conv import ConvLayer
from layers.flatten import FlattenLayer
from layers.maxpool import MaxPoolLayer
from layers.relu import ReluLayer
from layers.sequential import Sequential
from layers.softmax import SoftMaxLayer
from layers.full import FullLayer
from layers.cross_entropy import CrossEntropyLayer
from layers.dataset import fer2013
import numpy as np

(train_x, train_y), (val_x,val_y), (test_x, test_y) = fer2013()
#Since network is extremely slow, training on the first 1500 images
train_x = train_x[0:1500,:,:,:]

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
print("Initiating training")
loss = myNet.fit(x=train_x,y=train_y,epochs=epochs,lr=lr,batch_size=batch_size)
myNet.save()
pred = myNet.predict(val_x)
accuracy = np.mean(pred == val_y)
print('At learning rate = '+str(lr))
print('Accuracy of Convolutional Neural Network = '+str(accuracy))


