import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

while(True): # process individual frames of video
    ret, img = cap.read()

    # Indicate where to place hand and make that place a region of interest
    cv2.rectangle(img, (100, 50), (550, 550), (255, 0, 0), 0)
    img_roi = img[50:550, 100:550] # Crop out rectangle w/ hand

    # apply transformations to make reading hand easier/cleaner
    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY) # convert to gray
    blurred = cv2.GaussianBlur(gray, (15, 15), 0) # apply gaussian blur
    ret, thresh = cv2.threshold(blurred, 100, 255, # apply threshold for B/W
                                cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Get contours of hand
    img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                                 cv2.CHAIN_APPROX_SIMPLE)

    # Get and draw largest contour (one around hand) and hull around hand
    largest_contour = max(contours, key =  lambda x: cv2.contourArea(x))
    hull = cv2.convexHull(largest_contour)
    # Calculate center of contour and draw circle at centroid
    moment = cv2.moments(largest_contour)
    cX = int(moment['m10']/moment['m00'])
    cY = int(moment['m01']/moment['m00'])
    cv2.drawContours(img_roi, [largest_contour], 0, (0, 255, 0), 2)
    cv2.drawContours(img_roi, [hull], 0, (0, 0, 255), 2)
    cv2.circle(img_roi, (cX, cY), 5, (0, 255, 255), 2)
    cv2.circle(img_roi, (cX, cY), 50, (0, 128, 255), 2)

    # Get defects for tracking fingertips
    hull2 = cv2.convexHull(largest_contour, returnPoints = False)
    defects = cv2.convexityDefects(largest_contour, hull2)
    count = 0 # Track total number of fingertips registered
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(largest_contour[s][0]) # tip
        end = tuple(largest_contour[e][0]) # between
        far = tuple(largest_contour[f][0]) # tip

        # Find angle of finger using defects and cosine rule (given 3 sides)
        sidex = math.sqrt(math.pow((start[0] - end[0]), 2)
                             + math.pow((start[1] - end[1]), 2))
        sidey = math.sqrt(math.pow((start[0] - far[0]), 2)
                             + math.pow((start[1] - far[1]), 2))
        sidez = math.sqrt(math.pow((far[0] - end[0]), 2)
                             + math.pow((far[1] - end[1]), 2))
        arc_angle = ((math.pow(sidex, 2) - math.pow(sidey, 2)
                      - math.pow(sidez, 2)) / (-2 * sidey * sidez))
        angle = math.acos(arc_angle) * 180 / math.pi # Calculate & convert

        # Find angle between centroid and fingertip to further filter
        center_angle = math.atan2(cY - end[1], cX - end[0]) * 180 / math.pi

        # Fingertip must be a certain distance from centroid (filtering)
        (x, y, w, h) = cv2.boundingRect(largest_contour)
        if angle <= 80 and sidey > 0.25 * h and center_angle <= 150:
            count += 1 # Increment number of fingertips
            cv2.circle(img_roi, end, 5, (0, 0, 255), 3) # Draw on fingertip

    cv2.imshow('Image', img)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
