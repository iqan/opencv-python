import numpy as np
import cv2
import cv2.cv as cv

cap = cv2.VideoCapture(0)
fgbg = cv2.BackgroundSubtractorMOG(30,3,0.7)
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
scale_down = 4

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(frame)
    #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    #test CONTOURS
    ret,thresh = cv2.threshold(fgmask,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    #if len(contours) > 0:
     #   m = np.mean(contours[0],axis=0)
     #   #print 'm[0,0] = ',m[0,0],'||  m[0,1] = ',m[0,1]

   # largestContour = contours[0]
  #  maxArea = 0;
   # for contour in enumerate(contours):
    #    area = cv2.contourArea(contour)
     #   if area > maxArea:
      #      maxArea = area
      #      largestContour = contour
    largest_contour = contours
    max_area = 0
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour
            moment = cv2.moments(largest_contour[0])
            if moment["m00"] > 1000 / scale_down:
                rect = cv2.minAreaRect(largest_contour)
                rect = ((rect[0][0] * scale_down, rect[0][1] * scale_down), (rect[1][0] * scale_down, rect[1][1] * scale_down), rect[2])
                box = cv2.cv.BoxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(frame,[box], 0, (0, 0, 255), 2)          
    #cv2.drawContours(frame, largest_contour, -1, (0,255,0), 3)
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('masked-frame',fgmask)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
