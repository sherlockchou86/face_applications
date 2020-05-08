


import cv2
import time
import face_recognition

cap = cv2.VideoCapture(0)
index = 0
start = 0
while cap.isOpened:
    if index == 0:
        start = time.time()
    
    ret, frame = cap.read()
    face_locations = face_recognition.face_locations(frame, model='cnn')

    for top, right, bottom, left in face_locations:

        # Extract the region of the image that contains the face
        face_image = frame[top:bottom, left:right]

        # Blur the face image
        face_image = cv2.GaussianBlur(face_image, (99, 99), 30)

        # Put the blurred face region back into the frame image
        frame[top:bottom, left:right] = face_image

    cv2.imshow('frame', frame)
    index = index + 1

    if cv2.waitKey(1) & 0xff == ord("q"):
        break
    
    if index == 10:
        index = 0
        end = time.time()
        print('FPS: {}'.format(1 / ((end - start) / 10)))
