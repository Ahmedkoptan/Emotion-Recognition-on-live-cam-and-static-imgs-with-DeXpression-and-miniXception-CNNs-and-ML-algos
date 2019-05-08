import numpy as np
import pandas as pd
import os

root_dir = os.path.abspath('../Dataset')
data_dir = os.path.join(root_dir,'fer2013')


def fer2013():
    """
    Load fer2013 data
    """

    data = pd.read_csv(os.path.join(root_dir,data_dir,'fer2013.csv'))
    train = data[data['Usage'] == 'Training']
    test = data[data['Usage'] == 'PublicTest']
    val = data[data['Usage'] == 'PrivateTest']
    del(data)
    train_x = np.array([])
    test_x = np.array([])
    val_x = np.array([])
    train_y = np.array([])
    test_y = np.array([])
    val_y = np.array([])

    for i in range(len(train)):
        train_x = np.append(train_x, np.array((train.iloc[i]['pixels']).split(" "), dtype=np.int16)) 
        train_y = np.append(train_y, int(train.iloc[i]['emotion']))
    print("Training data ready")
    
    del(train)

    for i in range(len(test)):
        test_x = np.append(test_x, np.array((test.iloc[i]['pixels']).split(" "), dtype=np.int16)) 
        test_y = np.append(test_y, int(test.iloc[i]['emotion']))
    print("Testing data ready")
    
    del(test)

    for i in range(len(val)):
        val_x = np.append(val_x, np.array((val.iloc[i]['pixels']).split(" "), dtype=np.int16)) 
        val_y = np.append(val_y, int(val.iloc[i]['emotion']))
    print("Validation data ready")
    
    del(val)
    """
    train_x = np.asarray(train_x)
    val_x = np.asarray(val_x)
    test_x = np.asarray(test_x)
    """

    """
    train_y = onehot(train_y)
    val_y = onehot(val_y)
    test_y = onehot(test_y)
    
    train_y = np.asarray(train_y)
    val_y = np.asarray(val_y)
    test_y = np.asarray(test_y)
    
    
    train_x = scale(train_x)
    val_x = scale(val_x)
    test_x = scale(test_x)
    """

    """
    train_x = train_x.reshape((train_x.shape[0], 48, 48))
    val_x = val_x.reshape((val_x.shape[0], 48, 48))
    test_x = test_x.reshape((test_x.shape[0], 48, 48))
    """

    return (train_x, train_y), (val_x,val_y), (test_x, test_y)



def onehot(y):
    """
    Transform vector into one-hot representation
    """
    y_oh = np.zeros((len(y), np.max(y)+1))
    y_oh[np.arange(len(y)), y] = 1
    return y_oh


def scale(x):
    """
    Scale data to be between -0.5 and 0.5
    """
    x = x.astype('float') / 255.
    x = x - 0.5
    x = x.reshape(-1, 48*48)
    return x
