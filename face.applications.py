
import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time

EAR_THRESHOLD = 0.3
ENCODING_DISTANCE = 0.5

# need modify according to fps
# fps is about 15 when set below
BLINK_FRAMES_THRESHOLD = 2
CLOSE_EYE_FRAMES_THRESHOLD = 15
NOT_ALIVE_FRAMES_THRESHOLD = 75

COUNTER = 0
BLINK_COUNTER = 0
LAST_BLINK_FRAME_INDEX = 0

# 4 types of alarms
CLOSE_EYE_ALARM = False
ABSENCE_ALARM = False
UNKNOWN_PERSON_ALARM = False
NOT_ALIVE_ALARM = False


# get EAR from eye landmark
def get_ear(eye_landmark):
    eye_landmark = np.array(eye_landmark)
    A = np.sqrt(np.sum(np.square(eye_landmark[1]-eye_landmark[5])))  
    B = np.sqrt(np.sum(np.square(eye_landmark[2]-eye_landmark[4])))  
    C = np.sqrt(np.sum(np.square(eye_landmark[0]-eye_landmark[3]))) 

    return (A + B) / (2.0 * C)


# draw text on screen
def putText(frame, text, color, location, size=20):
  cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  pil_im = Image.fromarray(cv2_im)

  draw = ImageDraw.Draw(pil_im)
  font = ImageFont.truetype("./fonts/msyh.ttc", size, encoding="utf-8")
  draw.text(location, text, color, font=font)

  cv2_text_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

  return cv2_text_im


# load the right person
# we will do face compare later
the_right_person = cv2.imread('./the_right_person.jpg')
the_right_person_location = face_recognition.face_locations(the_right_person)[0]
the_right_person_encoding = face_recognition.face_encodings(the_right_person)[0]
# crop the face
the_right_person = the_right_person[(the_right_person_location[0] - 10):(the_right_person_location[2] + 10), (the_right_person_location[3] - 10):(the_right_person_location[1] + 10)]
the_right_person = putText(the_right_person, '目标人脸', (0, 255, 255), (10, 10), size=12)
#print(the_right_person_encoding)


# capture video camera
cap = cv2.VideoCapture(0)
index = 0

# loop start
while cap.isOpened:
    start = time.time()

    ret, frame = cap.read()
    
    # get face locations, use GPU
    face_locations = face_recognition.face_locations(frame, model='cnn')
    # get face landmarks
    face_landmarks = face_recognition.face_landmarks(frame, face_locations)
    # get face encodings
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    # draw face locations
    for face_rect in face_locations:
        #print(face_rect)
        frame = cv2.rectangle(frame, (face_rect[1], face_rect[0]), (face_rect[3], face_rect[2]), (255, 0, 0), 2)
    
    # draw face landmarks
    for face_landmark in face_landmarks:
        #print(face_landmark)
        for face_landmark_key in face_landmark.keys():
            pts = face_landmark[face_landmark_key]
            for i in range(len(pts)-1):
                cv2.line(frame, pts[i], pts[i+1],(255, 0, 0))
            for pt in pts:
                cv2.circle(frame, pt, 2, (255, 0, 0), -1)
    

    # alarm logic
    # we only deal with the first face in frame
    if len(face_locations) == 0:
        ABSENCE_ALARM = True
        UNKNOWN_PERSON_ALARM = True
        NOT_ALIVE_ALARM = False
        CLOSE_EYE_ALARM = False
        BLINK_COUNTER = 0
        COUNTER = 0
        LAST_BLINK_FRAME_INDEX = 0
    else:
        ABSENCE_ALARM = False
        first_face_location = face_locations[0]
        first_face_landmark = face_landmarks[0]
        first_face_encoding = face_encodings[0]

        # check if the person is the right person
        face_distances = face_recognition.face_distance([the_right_person_encoding], first_face_encoding)
        print('face distance: {}'.format(face_distances[0]))
        if face_distances[0] > ENCODING_DISTANCE:
            UNKNOWN_PERSON_ALARM = True
        else:
            UNKNOWN_PERSON_ALARM = False

        left_ear = get_ear(first_face_landmark["left_eye"])
        right_ear = get_ear(first_face_landmark["right_eye"])

        # we use average value
        ear = (left_ear + right_ear) / 2.0
        print('EAR: {}'.format(ear))

        if ear < EAR_THRESHOLD:
            COUNTER = COUNTER + 1
            if COUNTER > CLOSE_EYE_FRAMES_THRESHOLD:
                CLOSE_EYE_ALARM = True
        else:
            if COUNTER > BLINK_FRAMES_THRESHOLD:
                BLINK_COUNTER = BLINK_COUNTER + 1
                LAST_BLINK_FRAME_INDEX = index
            
            CLOSE_EYE_ALARM = False
            COUNTER = 0

        if index - LAST_BLINK_FRAME_INDEX > NOT_ALIVE_FRAMES_THRESHOLD:
            NOT_ALIVE_ALARM = True
        else:
            NOT_ALIVE_ALARM = False
        
    
    index = index + 1   
    
    
    # alarm notify on screen
    frame[0:the_right_person.shape[0], 0:the_right_person.shape[1]] = the_right_person

    if ABSENCE_ALARM:
        frame = putText(frame, '座位：无人', (255, 0, 0), (frame.shape[1]-120, 10), size=15)
    else:
        frame = putText(frame, '座位：有人', (0, 255, 255), (frame.shape[1]-120, 10), size=15)
    
    if UNKNOWN_PERSON_ALARM:
        frame = putText(frame, '授权：失败', (255, 0, 0), (frame.shape[1]-120, 30), size=15)
    else:
        frame = putText(frame, '授权：成功', (0, 255, 255), (frame.shape[1]-120, 30), size=15)
    
    frame = putText(frame, '眨眼：{}'.format(BLINK_COUNTER), (0, 255, 255), (frame.shape[1]-120, 50), size=15)
    
    if NOT_ALIVE_ALARM:
        frame = putText(frame, '活体：否/请眨眼', (255, 0, 0), (frame.shape[1]-120, 70), size=15)
    elif not ABSENCE_ALARM:
        frame = putText(frame, '活体：是', (0, 255, 255), (frame.shape[1]-120, 70), size=15)
    
    if CLOSE_EYE_ALARM:
        frame = putText(frame, '疲劳：是/请睁眼', (255, 0, 0), (frame.shape[1]-120, 90), size=15)
    elif not ABSENCE_ALARM:
        frame = putText(frame, '疲劳：否', (0, 255, 255), (frame.shape[1]-120, 90), size=15)
    

    # show frame
    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break
    
    end = time.time()
    print('FPS: {}'.format(1 / (end - start)))

cv2.destroyAllWindows()