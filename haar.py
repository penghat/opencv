import cv2
import numpy as np

hand_cascade = cv2.CascadeClassifier('cascade-index.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    hands = hand_cascade.detectMultiScale(gray)
    print(len(hands))
    for (x, y, w, h) in hands:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('img', gray)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
