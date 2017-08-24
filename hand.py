import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True): # process individual frames of video
    ret, img = cap.read()

    # Indicate where to place hand and make that place a region of interest
    cv2.rectangle(img, (100, 50), (550, 550), (255, 0, 0), 4)
    img_roi = img[50:550, 100:550] # Crop out rectangle w/ hand

    # apply transformations to make reading hand easier/cleaner
    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY) # convert to gray
    blurred = cv2.GaussianBlur(gray, (15, 15), 0) # apply gaussian blur
    ret, thresh = cv2.threshold(blurred, 100, 255, # apply threshold for B/W
                                cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    cv2.imshow('ROI', img_roi)
    cv2.imshow('After transformation', thresh)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
