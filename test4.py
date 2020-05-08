
'''
face detect test for MTCNN
'''

from mtcnn import MTCNN
import cv2

mt = MTCNN()

image = cv2.imread('./face7.jpg')

results = mt.detect_faces(image)

for face in results:
    cv2.rectangle(image, (face['box'][0], face['box'][1]), (face['box'][0] + face['box'][2], face['box'][1] + face['box'][3]), (255, 0, 0), 2)

cv2.imshow('result', image)
k = cv2.waitKey(0) 
if k ==27:     
    cv2.destroyAllWindows() 
