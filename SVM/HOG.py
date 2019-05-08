import scipy
import numpy as np
from skimage.feature import hog


def sliding_hog_windows(image,image_height,image_width,window_size,window_step):
    hog_windows = []
    for y in range(0, image_height, window_step):
        for x in range(0, image_width, window_step):
            window = image[y:y + window_size, x:x + window_size]
            hog_windows.extend(hog(window, orientations=8, pixels_per_cell=(8, 8),
                                   cells_per_block=(1, 1), visualise=False))
    return hog_windows



def HOG(train_x,val_x,test_x):

    hog_features = []
    hog_images = []
    image_height = 48
    image_width = 48
    window_size = 24
    window_step = 6
    for i in range(len(train_x)):
        image=train_x[i].reshape((image_height, image_width))

        # HOG windows features
        features=sliding_hog_windows(image,image_height,image_width,window_size,window_step)
        f, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                           cells_per_block=(1, 1), visualise=True)
        hog_features.append(features)
        hog_images.append(hog_image)

        print("HOG running: ")
    Train_x = hog_features


    hog_features = []
    hog_images = []
    for i in range(len(val_x)):
        image = val_x[i].reshape((image_height, image_width))

        # HOG windows features
        features = sliding_hog_windows(image, image_height, image_width, window_size, window_step)
        f, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                           cells_per_block=(1, 1), visualise=True)
        hog_features.append(features)
        hog_images.append(hog_image)

        print("HOG running: ")
    Val_x = hog_features


    hog_features = []
    hog_images = []
    for i in range(len(test_x)):
        image = test_x[i].reshape((image_height, image_width))

        # HOG windows features
        features = sliding_hog_windows(image, image_height, image_width, window_size, window_step)
        f, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                           cells_per_block=(1, 1), visualise=True)
        hog_features.append(features)
        hog_images.append(hog_image)
        print("HOG running: ")
    Test_x = hog_features

    return Train_x, Val_x, Test_x