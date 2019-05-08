import scipy
import numpy as np
import cv2
import scipy.misc
import dlib

def get_landmarks(image, rects,predictor):
    # this function have been copied from http://bit.ly/2cj7Fpq
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

def Landmarks(train_x,val_x,test_x):
    landmarks = []

    image_height = 48
    image_width = 48
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    for i in range(len(train_x)):
        image = train_x[i].reshape((image_height, image_width))

        scipy.misc.imsave('temp.jpg', image)
        image2 = cv2.imread('temp.jpg')
        face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
        face_landmarks = get_landmarks(image2, face_rects,predictor)
        landmarks.append(face_landmarks)

    Train_x=landmarks
    #print("before flatten():")
    #print(np.shape(Train_x))
    Train_x=np.array([x.flatten() for x in Train_x])
    #print("after flatten():")

    #print(np.shape(Train_x))
    Train_x = np.squeeze(Train_x, axis=1)
    #print("after squeeze:")
    #print(np.shape(Train_x))
    #print(Train_x)

    landmarks = []

    for i in range(len(val_x)):
        image = val_x[i].reshape((image_height, image_width))

        scipy.misc.imsave('temp.jpg', image)
        image2 = cv2.imread('temp.jpg')
        face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
        face_landmarks = get_landmarks(image2, face_rects,predictor)
        landmarks.append(face_landmarks)

    Val_x=landmarks
    Val_x = np.array([x.flatten() for x in Val_x])
    Val_x = np.squeeze(Val_x, axis=1)

    landmarks = []

    for i in range(len(test_x)):
        image = test_x[i].reshape((image_height, image_width))

        scipy.misc.imsave('temp.jpg', image)
        image2 = cv2.imread('temp.jpg')
        face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
        face_landmarks = get_landmarks(image2, face_rects, predictor)
        landmarks.append(face_landmarks)

    Test_x = landmarks
    Test_x = np.array([x.flatten() for x in Test_x])
    Test_x = np.squeeze(Test_x, axis=1)
    return Train_x, Val_x, Test_x