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

# Give a run ID here. Change it to flags (arguments) in version 2.
ID = '4_3'
RUNID = 'DeXpression_run_' + ID

# Give a dropout if required (change to True and define the dropout percentage).
do = True
dropout_keep_prob=0.5
LR = 0.0001


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

# Save the model
#model.save('../DeXpression/DeXpression_checkpoints/' + RUNID + '.model')

model.load('../DeXpression/DeXpression_checkpoints/' + RUNID + '.model')


EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]


import cv2
import sys
import imutils
from keras.preprocessing.image import img_to_array


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

cv2.namedWindow('your_face')
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = imutils.resize(frame, width=400)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    # Draw a rectangle around the faces
    if len(faces)>0:
        faces = sorted(faces, reverse=True,
                       key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        fx,fy,fw,fh = faces
        roi = gray[fy:fy+fh,fx:fx+fw]
        roi = cv2.resize(roi,(48,48))
        roi = roi.astype(float)/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi)[0]
        prediction = np.argmax(preds)
        emotion_probability = np.max(preds)
        label = EMOTIONS[prediction]

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            # construct the label text
            text = "{}: {:.2f}%".format(emotion, prob * 100)
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5),
                          (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 2)
            cv2.putText(frameClone, label, (fx, fy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fx, fy), (fx + fw, fy + fh),
                          (0, 0, 255), 2)

    cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

