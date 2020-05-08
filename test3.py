import cv2
import time
import face_recognition


image = cv2.imread('./face14.jpg')

face_locations = face_recognition.face_locations(image, model='cnn', number_of_times_to_upsample=1)

for face_rect in face_locations:
    image = cv2.rectangle(image, (face_rect[1], face_rect[0]), (face_rect[3], face_rect[2]), (255, 0, 0), 2)


cv2.imshow('result', image)
k = cv2.waitKey(0) 
if k ==27:     
    cv2.destroyAllWindows() 