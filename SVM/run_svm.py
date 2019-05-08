import sklearn .svm
import numpy as np
from dataset import fer2013
#from PCA import PCA_dim_red
import numpy as np
import scipy
import os
from sklearn.model_selection import GridSearchCV
import pickle

def PCA_dim_red(train_x,val_x,test_x,var):
    # Dimensionality Reduction
    m = train_x.shape[0]
    n = train_x.shape[1]

    Mu = np.mean(train_x, axis=0)
    print(Mu.shape)
    train_x = train_x - Mu

    Sigma = (train_x.T).dot(train_x) / (m - 1)
    U, S, V = np.linalg.svd(Sigma)

    # first 253 components maintain 95% variance
    # first 881 components maintain 99% variance
    tr = 0
    k=1
    while tr < var:
        tr = np.sum(S[:k])/np.sum(S)
        k+=1
    print('Using k = '+str(k)+', '+str(tr)+' of the variance was retained')
    V = V[:,:k]
    Train_x = train_x.dot(V)
    Val_x = val_x.dot(V)
    Test_x = test_x.dot(V)
    return Train_x,Val_x,Test_x

#import matplotlib
#matplotlib.use('Agg')

#import matplotlib.pyplot as plt

#Setting directories
root_dir = os.path.abspath('../FER2013')
svm_dir = os.path.join(root_dir,'SVM')

#Importing Data
print("Importing data")
(train_x, train_y), (val_x,val_y), (test_x, test_y) = fer2013()
print("Imported data, initiating training")
#train_x = train_x[0:20000,:]
#train_y = train_y[0:20000]
#originally (28709, 2304)




#######SVMs without Dimensionality reduction

#Model Selection
parameters = {'kernel':('linear', 'poly', 'rbf'), 'C':[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]}
svc = sklearn.svm.SVC()
clf = GridSearchCV(svc, parameters, cv=5)
Train_x = np.concatenate((train_x, val_x), axis=0)
Train_y = np.concatenate((train_y,val_y), axis=0)
clf.fit(Train_x,Train_y)
sorted(clf.cv_results_.keys())

# save the model to disk
filename = 'svm.sav'
pickle.dump(clf, open(filename, 'wb'))
"""
# load the model from disk
loaded_model = pickle.load(open('svm.sav', 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)
"""

"""
SVM = sklearn.svm.SVC(C=10.0, kernel='linear')
SVM.fit(train_x, train_y)
y_pred = SVM.predict(val_x)
lin_accuracy = np.mean(y_pred == val_y)
print("SVM accuracy with linear kernel = " + str(lin_accuracy))

Poly_SVM = sklearn.svm.SVC(C=1.0, kernel='poly', coef0=1.0)
Poly_SVM.fit(train_x, train_y)
y_pred = Poly_SVM.predict(val_x)
poly_accuracy = np.mean(y_pred == val_y)
print("SVM accuracy with polynomial kernel= " + str(poly_accuracy))

Gaussian_SVM = sklearn.svm.SVC(C=9.0, kernel='rbf', gamma='scale')
Gaussian_SVM.fit(train_x, train_y)
y_pred = Gaussian_SVM.predict(val_x)
rbf_accuracy = np.mean(y_pred == val_y)
print("SVM accuracy with Gaussian kernel= " + str(rbf_accuracy))
"""






########SVMs with Dimensionality Reduction

#After 99% variance 881 components, 95% variance 253 components
var = 0.99
Train_x,Val_x,Test_x = PCA_dim_red(train_x,val_x,test_x,var)

#Model Selection
parameters = {'kernel':('linear', 'poly', 'rbf'), 'C':[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]}
svc = sklearn.svm.SVC()
clf = GridSearchCV(svc, parameters, cv=5)
Train_x = np.concatenate((Train_x, Val_x), axis=0)
clf.fit(Train_x,Train_y)
sorted(clf.cv_results_.keys())

# save the model to disk
filename = 'svm_pca_99.sav'
pickle.dump(clf, open(filename, 'wb'))
"""
#To Load the model later for predictions
# load the model from disk
loaded_model = pickle.load(open('svm_pca.sav', 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)
"""

"""
SVM = sklearn.svm.SVC(C=10.0, kernel='linear')
SVM.fit(Train_x, train_y)
y_pred = SVM.predict(Val_x)
#Store the weights
#output the accuracy
lin_accuracy = np.mean(y_pred == val_y)
print("SVM accuracy with linear kernel at C = "+str(C[i])+" is equal to " + str(lin_accuracy))

Poly_SVM = sklearn.svm.SVC(C=1.0, kernel='poly', coef0=1.0)
Poly_SVM.fit(Train_x, train_y)
y_pred = Poly_SVM.predict(Val_x)
poly_accuracy = np.mean(y_pred == val_y)
print("SVM accuracy with polynomial kernel at C = "+str(C[i])+" is equal to " + str(poly_accuracy))

Gaussian_SVM = sklearn.svm.SVC(C=9.0, kernel='rbf', gamma='scale')
Gaussian_SVM.fit(Train_x, train_y)
# Store the weights
# output the accuracy
y_pred = Gaussian_SVM.predict(Val_x)
rbf_accuracy = np.mean(y_pred == val_y)
print("SVM accuracy with Gaussian kernel at C = "+str(C[i])+" is equal to " + str(rbf_accuracy))
"""

