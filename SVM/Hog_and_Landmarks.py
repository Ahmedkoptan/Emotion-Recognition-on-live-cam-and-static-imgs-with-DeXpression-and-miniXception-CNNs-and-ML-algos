from Landmarks import Landmarks
from HOG import HOG
import numpy as np




def Hog_and_Landmarks(train_x,val_x,test_x):
    Train_x1, Val_x1, Test_x1 = Landmarks(train_x, val_x, test_x)
    Train_x2, Val_x2, Test_x2 = HOG(train_x, val_x, test_x)
    print("Train_x1:")
    print(np.shape(Train_x1))
    print("Train_x2:")
    print(np.shape(Train_x2))
    Train_x3 = np.concatenate((Train_x1,Train_x2), axis=1)
    print("Train_x3:")
    print(np.shape(Train_x3))
    Val_x3 = np.concatenate((Val_x1, Val_x2), axis=1)
    Test_x3 = np.concatenate((Test_x1, Test_x2), axis=1)

    return Train_x3, Val_x3, Test_x3