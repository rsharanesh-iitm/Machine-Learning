import cv2
webcam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    successfull_frame_read, frame = webcam.read()
    if not successfull_frame_read:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector_default.detectMultiScale(frame_gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (100,200,50), 4)

    cv2.imshow("Face_detector_default",frame)
    cv2.waitKey(1)
