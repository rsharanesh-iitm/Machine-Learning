# Importing the libraries
import cv2
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image

# Loading the haarcascade classifier for face detection
cascPath = "/home/sharaneh/Downloads/haar_cascade_frontal_face.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Loading the pretraind classifier model
tl_model = keras.models.load_model("tl_model_params.h5")
cnn_model = keras.models.load_model("tl_model_params.h5")
classes = ['bhuvneshwar_kumar',
 'dinesh_karthik',
 'hardik_pandya',
 'jasprit_bumrah',
 'k._l._rahul',
 'kedar_jadhav',
 'kuldeep_yadav',
 'mohammed_shami',
 'ms_dhoni',
 'ravindra_jadeja',
 'rohit_sharma',
 'shikhar_dhawan',
 'vijay_shankar',
 'virat_kohli',
 'yuzvendra_chahal']

# Starting the camera
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame video
    ret, frame = video_capture.read()

    # Converting to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting the faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30))

    # Draw a rectangle around the faces and annotating the name of the player
    for (x, y, w, h) in faces:
        #Drwaing the rectangles
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        img1 = frame[y:y+h,x:x+w] #Slecting the face part of the video alone
        img2 = cv2.resize(img1,(256,256)) #Resizing the image to the model trained image size

        tl_pred = tl_model.predict(img2.reshape(1,256,256,3)) #Predicting the best player the face resembles to
        tl_top_match = np.argmax(tl_pred)
        cnn_pred = cnn_model.predict(img2.reshape(1,256,256,3)) #Predicting the best player the face resembles to
        cnn_top_match = np.argmax(cnn_pred)

        # Annotating the frame with the name of the player it resembles o
        cv2.putText(frame, str(classes[tl_top_match]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.putText(frame, str(classes[cnn_top_match]), (x+w, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()