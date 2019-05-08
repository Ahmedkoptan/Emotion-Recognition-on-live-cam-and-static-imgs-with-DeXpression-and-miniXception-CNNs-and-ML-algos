This project directory contains machine learning and deep learning models that are used to predict emotion from facial expression of live webcam videos and static images. The models are trained on Fer2013 Training subset (almost 28k samples), and validated performance on Public Testing subset (almost 3k samples). There is a testing subset. The Testing subset is composed of 3k samples, and they can be used to test the final validated model performance for generalization error.
The Fer2013 dataset contains samples that are grayscale face entered images. Each image is 48 x 48 pixels.


The dataset was downloaded from Kaggle : 
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
Please replace the folder in the Dataset directory with the extracted folder (highlighted with * below). 
The final Dataset directory should be as follows:
+root/project/directory
  +Dataset
    +fer2013*
      +idea
      fer2013.bib
      fer2013.csv
      README


The models included here are:
1) Convolutional Neural Network (CNN) made from scratch (using numpy and script libraries) Python 2.7 
2) mini Xception (Python 3.6)
3) DeXpression (Python 2.7)*
4) Support Vector Machine (SVM with multiple kernels) (Python 2.7)
5) Gaussian Naive Bayes (GNB) (Python 2.7)
6) Logistic Regression (LR) (Python 2.7)

Only DeXpression and mini Xception have been used in the live webcam video

1) The CNN directory contains the network made from scratch. 
The file run_fer2013.py can be used to train the CNN
The file pred_cnn.py can be used to test the network on static images (whether from validation set or testing set)
The file loss.txt contains the saved model weights
The file mycnnscript.sh is a shell file used to submit batch submissions to Agave cluster
The layers directory contains the definition of the CNN layers made from scratch

2) The Xception directory contains the mini Xception network. The code in this directory was written in Python3
The file FaceEmotion.py can be used to train and test the network on static images and live webcam videos
The files in this directory were originally downloaded from :"https://github.com/abhijeet3922/FaceEmotion_ID" and modified accordingly

3) The DeXpression directory contains the DeXpression network.
The file DeXpression.py is used to train the network
The file dataset.py loads the data
The file pred.py predicts the performance on validation and testing sets
The file detect.py uses the trained model to predict on live video (this can be run using either python 2.7 or 3.7)
Will add layer l2 regularization in future updates to prevent overfitting. For now, when training, you can use a high drop out and an initial learning rate around 0.001

4) The SVM directory contains the SVM model
The file run_svm.py trains the SVM using Linear, Polynomial, and Gaussian kernels on training set and does grid search for best C parameter on validation set. Then the weights are saved on disk to be used later for evaluation on testing set. There are 3 models. The first model uses no dimensionality reduction. The second model uses Landmarks to extract important pixels then PCA to reduce the dimensions of the model while maintaining 99% variance. The third model uses HOG and Landmarks instead of PCA and Landmarks.
The file mysvmscript.sh is a shell script for a batch submission on Agave cluster
The file dataset.py is for reading the data

5) The GNB directory contains the Gaussian Naive Bayes model
The file run_gnb.py trains a GNB on training set and validates on validation set
Model saving and prediction on testing set will be implemented in a future update

6) The LR directory contains the Logistic Regression model
The run_lr.py file trains a logistic regression model using multiple Cs (for regularisation) and validates on the validation set
Model saving and prediction on testing set will be implemented in a future update

The files PCA.py and Landmarks.py are used as reference files for functions written in SVM and that will be written in GNB and LR in future updates

The video file is a sample test using a trained mini Xception network