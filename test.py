


import cv2
import time


cap = cv2.VideoCapture(0)
index = 0
start = 0
while cap.isOpened:
    if index == 0:
        start = time.time()
    
    ret, frame = cap.read()

    cv2.imshow('frame', frame)
    index = index + 1

    if cv2.waitKey(500) & 0xff == ord("q"):
        break
    
    if index == 10:
        index = 0
        end = time.time()
        print('FPS: {}'.format(1 / ((end - start) / 10)))
