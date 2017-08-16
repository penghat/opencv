import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, width, height) in faces: # draw rectangle for every face
        # where to draw, starting coord, end coord, color, width
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 255), 2)
        print((x, y))
         # = location of face 
        roi_gray = gray[y:y + height, x:x + width]
        roi_gray = img[y:y + height, x:x + width]

        # Draw the eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (eye_x, eye_y, eye_width, eye_height) in eyes:
            cv2.rectangle(roi_gray, (eye_x, eye_y), 
                          (eye_x + eye_width, eye_y + eye_height), 
                          (0, 255, 0), 2)


    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
        
    


