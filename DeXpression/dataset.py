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
    train_y = []
    train_x = []
    val_x = []
    val_y = []
    test_x = []
    test_y = []
    for i in range(data.shape[0]):
        if data.Usage[i] == 'Training':
            train_x.append(map(int,data.pixels[i].split(" ")))
            train_y.append(data.emotion[i])
        elif data.Usage[i] == 'PublicTest':
            val_x.append(map(int,data.pixels[i].split(" ")))
            val_y.append(data.emotion[i])
        else:
            test_x.append(map(int,data.pixels[i].split(" ")))
            test_y.append(data.emotion[i])

    train_x = np.asarray(train_x)
    val_x = np.asarray(val_x)
    test_x = np.asarray(test_x)


    train_y = onehot(train_y)
    val_y = onehot(val_y)
    test_y = onehot(test_y)

    train_x = scale(train_x)
    val_x = scale(val_x)
    test_x = scale(test_x)

    train_x = train_x.reshape((train_x.shape[0], 48, 48, 1))
    val_x = val_x.reshape((val_x.shape[0], 48, 48, 1))
    test_x = test_x.reshape((test_x.shape[0], 48, 48, 1))

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
