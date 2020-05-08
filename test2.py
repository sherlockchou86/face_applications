import cv2
import time
import face_recognition
from PIL import Image, ImageDraw
import numpy as np

cap = cv2.VideoCapture(0)
index = 0
start = 0
while cap.isOpened:
    if index == 0:
        start = time.time()
    
    ret, frame = cap.read()
    face_locations = face_recognition.face_locations(frame, model='cnn')
    face_landmarks_list = face_recognition.face_landmarks(frame, face_locations=face_locations)

    frame = Image.fromarray(frame)
    d = ImageDraw.Draw(frame, 'RGBA')

    for face_landmarks in face_landmarks_list:
        # Make the eyebrows into a nightmare
        d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
        d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
        d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
        d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

        # Gloss the lips
        d.polygon(face_landmarks['top_lip'], fill=(0, 0, 255, 128))
        d.polygon(face_landmarks['bottom_lip'], fill=(0, 0, 255, 128))
        d.line(face_landmarks['top_lip'], fill=(0, 0, 255, 64), width=8)
        d.line(face_landmarks['bottom_lip'], fill=(0, 0, 255, 64), width=8)

        # Sparkle the eyes
        d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
        d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

        # Apply some eyeliner
        d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
        d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

        d.line(face_landmarks['chin'], fill=(68, 54, 39, 255), width=8)
        d.polygon([face_landmarks['nose_bridge'][0]] + [face_landmarks['nose_tip'][0]] + [face_landmarks['nose_tip'][4]], fill=(68, 54, 39, 128))
    
    frame = np.array(frame)
    cv2.imshow('frame', frame)
    index = index + 1

    if cv2.waitKey(1) & 0xff == ord("q"):
        break
    
    if index == 10:
        index = 0
        end = time.time()
        print('FPS: {}'.format(1 / ((end - start) / 10)))