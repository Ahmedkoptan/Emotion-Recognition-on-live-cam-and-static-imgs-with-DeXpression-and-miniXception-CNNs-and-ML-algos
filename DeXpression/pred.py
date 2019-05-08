import numpy as np
import tflearn
import tflearn.activations as activations
# Data loading and preprocessing
from tflearn.activations import relu
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.conv import avg_pool_2d, conv_2d, max_pool_2d
from tflearn.layers.core import dropout, flatten, fully_connected, input_data
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.normalization import batch_normalization
from dataset import fer2013

# Give a run ID here. Change it to flags (arguments) in version 2.
ID = '4_3'
RUNID = 'DeXpression_run_' + ID

# Give a dropout if required (change to True and define the dropout percentage).
do = True
dropout_keep_prob=0.5
LR = 0.001

# Load data
(train_x, train_y), (val_x,val_y), (test_x, test_y) = fer2013()


# Define number of output classes.
num_classes = 7

# Define padding scheme.
padding = 'VALID'

# Model Architecture
network = input_data(shape=[None, 48, 48, 1])
#change stride, decrease filter size, same padding instead of normalization (this run)
conv_1 = relu(conv_2d(network, 64, 7, strides=2, bias=True, padding=padding, activation=None, name='Conv2d_1'))
maxpool_1 = batch_normalization(max_pool_2d(conv_1, 3, strides=2, padding=padding, name='MaxPool_1'))
#LRN_1 = local_response_normalization(maxpool_1, name='LRN_1')
# FeatEX-1
conv_2a = relu(conv_2d(maxpool_1, 96, 1, strides=1, padding=padding, name='Conv_2a_FX1'))
maxpool_2a = max_pool_2d(maxpool_1, 3, strides=1, padding=padding, name='MaxPool_2a_FX1')
conv_2b = relu(conv_2d(conv_2a, 208, 3, strides=1, padding=padding, name='Conv_2b_FX1'))
conv_2c = relu(conv_2d(maxpool_2a, 64, 1, strides=1, padding=padding, name='Conv_2c_FX1'))
FX1_out = merge([conv_2b, conv_2c], mode='concat', axis=3, name='FX1_out')
# FeatEX-2
conv_3a = relu(conv_2d(FX1_out, 96, 1, strides=1, padding=padding, name='Conv_3a_FX2'))
maxpool_3a = max_pool_2d(FX1_out, 3, strides=1, padding=padding, name='MaxPool_3a_FX2')
conv_3b = relu(conv_2d(conv_3a, 208, 3, strides=1, padding=padding, name='Conv_3b_FX2'))
conv_3c = relu(conv_2d(maxpool_3a, 64, 1, strides=1, padding=padding, name='Conv_3c_FX2'))
FX2_out = merge([conv_3b, conv_3c], mode='concat', axis=3, name='FX2_out')
net = flatten(FX2_out)
if do:
    net = dropout(net, dropout_keep_prob)
loss = fully_connected(net, num_classes,activation='softmax')

# Compile the model and define the hyperparameters
network = tflearn.regression(loss, optimizer='Adam',
                     loss='categorical_crossentropy',
                     learning_rate=LR)

# Final definition of model checkpoints and other configurations
model = tflearn.DNN(network, checkpoint_path='../DeXpression/DeXpression_checkpoints',
                    max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir="./tflearn_logs/")


model.load('../FER2013/DeXpression/DeXpression_checkpoints/' + RUNID + '.model')

pred = model.predict(val_x)
predictionval_y = np.argmax(pred,axis=1)
trueval_y = np.argmax(val_y,axis = 1)
accuracy = np.mean(predictionval_y==trueval_y)
print('At learning rate = '+str(LR))
print('Testing Accuracy of DeXpression = '+str(accuracy))

pred = model.predict(test_x)
predictiontest_y = np.argmax(pred,axis=1)
truetest_y = np.argmax(test_y,axis = 1)
accuracy = np.mean(predictiontest_y == truetest_y)
print('At learning rate = '+str(LR))
print('Testing Accuracy of DeXpression = '+str(accuracy))

