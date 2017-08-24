import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True): # process individual frames of video
    ret, img = cap.read()

    # apply transformations to make reading hand easier/cleaner
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to gray
    blurred = cv2.GaussianBlur(gray, (15, 15), 0) # apply gaussian blur
    ret, thresh = cv2.threshold(blurred, 0, 255,  # apply threshold for B/W
                                cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.imshow('Frame', thresh)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
