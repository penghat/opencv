import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Frame', gray)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()
