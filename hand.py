import cv2, numpy as np, math, random

def detect_hand():

    finger_count = [] # Holds the number of fingers raised
    finger_frames = 0  # Counts the number of times loop has run

    # Stuff needed for info/welcome messages to be printed to screen
    text = "Welcome to Computer Vision Rock-Paper-Scissors!"
    text2 = "Rock (0 fingers), paper (5 fingers), scissors (2 fingers)"
    text3 = "Your move:"
    text4 = "AI move:"
    text5 = "YOU WIN!"
    text6 = "YOU LOSE!"
    text7 = "YOU TIED!"
    font = cv2.FONT_HERSHEY_DUPLEX
    color = (255, 153, 255)

    # Image of rock/paper/scissors to be overlayed onto screen
    rockim = cv2.imread("rock.jpg")
    scissorim = cv2.imread("scissors.jpg")
    paperim = cv2.imread("paper.jpg")
    rock, paper, scissor = False, False, False
    ai_rock, ai_paper, ai_scissor = False, False, False

    cap = cv2.VideoCapture(0) # Capture from primary webcam

    while(True): # process individual frames of video

        ret, img = cap.read()

        # Draw info text on the screen
        cv2.putText(img, text, (270, 600), font, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(img, text2, (250, 650), font, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(img, text3, (700, 150), font, 1, color, 2, cv2.LINE_AA)
        cv2.putText(img, text4, (700, 400), font, 1, color, 2, cv2.LINE_AA)

        # Indicate where to place hand & make that place a region of interest
        cv2.rectangle(img, (100, 50), (550, 550), (255, 0, 0), 0)
        img_roi = img[50:550, 100:550] # Crop out rectangle w/ hand

        # apply transformations to make reading hand easier/cleaner
        gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY) # convert to gray
        blurred = cv2.GaussianBlur(gray, (15, 15), 0) # apply gaussian blur
        ret, thresh = cv2.threshold(blurred, 100, 255, # apply B/W threshold
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

            # Find angle of finger using defects and cosine rule
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
            center_angle = math.atan2(cY - end[1], cX - end[0]) * 180
            center_angle /= math.pi # Convert to degrees from radians

            # Fingertip must be a certain distance from centroid (filtering)
            (x, y, w, h) = cv2.boundingRect(largest_contour)
            if angle <= 80 and sidey > 0.25 * h and center_angle <= 150:
                count += 1 # Increment number of fingertips
                cv2.circle(img_roi, end, 5, (0, 0, 255), 3) # Draw on tips

            finger_count.append(count) # Add current count to list

            # Determine the actual number of fingers being held up
            if finger_frames == 1500: # Only operate every 'x' frames
                arr = np.array(finger_count) # convert to numpy array
                # Number of finger equals the most frequent value in array
                # Idea is that correct number should appear most often
                num_fingers = np.bincount(arr).argmax()
                if num_fingers == 0: # Rock move made
                    rock = True
                    paper = False
                    scissor = False
                elif num_fingers == 2: # Scissor move made
                    scissor = True
                    rock = False
                    paper = False
                elif num_fingers == 5: # Paper move made
                    paper = True
                    rock = False
                    scissor = False
                else:
                    print("Invalid move\n\n")

                ai_move = random.choice((0, 2, 5)) # Get computer move

                if ai_move == 0: # Computer move is rock
                    ai_rock = True
                    ai_paper = False
                    ai_scissor = False
                elif ai_move == 2: # Computer move is scissor
                    ai_scissor = True
                    ai_rock = False
                    ai_paper = False
                else: # Computer move is paper
                    ai_paper = True
                    ai_rock = False
                    ai_scissor = False

                finger_frames = 0  # Reset counter of frames
                del(finger_count[:]) # Reset array


            finger_frames += 1

        # Draw user move on screen using images
        if rock == True:
            img[50:250, 950:1150] = rockim  # Draw rock on screen
        elif paper == True:
            img[50:250, 950:1150] = paperim # Draw paper on screen
        elif scissor == True:
             img[50:250, 950:1150] = scissorim # Draw scissor on screen

        # Draw computer move on screen using images and result text
        if ai_rock == True:
            img[300:500, 950:1150] = rockim
            if rock == True: # Draw
                cv2.putText(img, text7, (550, 700), font, 1,
                            (255, 255, 102), 2, cv2.LINE_AA)
            elif paper == True: # Win for player
                cv2.putText(img, text5, (550, 700), font, 1,
                             (102, 255, 102),  2, cv2.LINE_AA)
            elif scissor == True: # Loss for player
                cv2.putText(img, text6, (550, 700), font, 1, (51, 51, 255),
                            2, cv2.LINE_AA)
        elif ai_paper == True:
            img[300:500, 950:1150] = paperim
            if paper == True: # Draw
                cv2.putText(img, text7, (550, 700), font, 1,
                            (255, 255, 102), 2, cv2.LINE_AA)
            elif scissor == True: # Win for player
                cv2.putText(img, text5, (550, 700), font, 1,
                             (102, 255, 102),  2, cv2.LINE_AA)
            elif rock == True: # Loss for player
                cv2.putText(img, text6, (550, 700), font, 1, (51, 51, 255),
                            2, cv2.LINE_AA)
        elif ai_scissor == True:
            img[300:500, 950:1150] = scissorim
            if scissor == True: # Draw
                cv2.putText(img, text7, (550, 700), font, 1,
                            (255, 255, 102), 2, cv2.LINE_AA)
            elif rock == True: # Win for player
                cv2.putText(img, text5, (550, 700), font, 1,
                             (102, 255, 102),  2, cv2.LINE_AA)
            elif paper == True: # Loss for player
                cv2.putText(img, text6, (550, 700), font, 1, (51, 51, 255),
                            2, cv2.LINE_AA)


        cv2.imshow('Image', img)

        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

detect_hand()
