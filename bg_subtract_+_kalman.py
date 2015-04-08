# backgroundSUbMOG
# findcontours -> largest
# get centroid by moments
# give centroids to kalman
# draw trajectories
# 8/4/15 1:13AM

import numpy as np
import cv2
import cv2.cv as cv
from kalman2d import Kalman2D


#INITS
cap = cv2.VideoCapture(0)
fgbg = cv2.BackgroundSubtractorMOG(30,3,0.7)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# These will get the trajectories for mouse location and Kalman estiamte
measured_points = []
kalman_points = []
# Create a new Kalman2D filter and initialize it with starting mouse location
kalman2d = Kalman2D()

#SOME FUNCTIONS
def drawCross(img, center, r, g, b):
    '''
    Draws a cross a the specified X,Y coordinates with color RGB
    '''

    d = 5
    t = 2

    color = (r, g, b)

    ctrx = center[0]
    ctry = center[1]

    cv2.line(img, (ctrx - d, ctry - d), (ctrx + d, ctry + d), color, t, cv2.CV_AA)
    cv2.line(img, (ctrx + d, ctry - d), (ctrx - d, ctry + d), color, t, cv2.CV_AA)


def drawLines(img, points, r, g, b):
    '''
    Draws lines 
    '''

    cv2.polylines(img, [np.int32(points)], isClosed=False, color=(r, g, b))


#------MAIN LOOP -----
#---------------------
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

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
    cx = 0
    cy = 1
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour
            #finding centroid
            M = cv2.moments(largest_contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            #print '(',cx,',',cy,')'

    # Serve up a fresh image
    img = np.zeros((500,500,3), np.uint8)

    # Grab current centroid and add it to the trajectory
    measured = (cx, cy)
    measured_points.append(measured)

    # Update the Kalman filter with the centroid
    kalman2d.update(cx, cy)

    # Get the current Kalman estimate and add it to the trajectory
    estimated = [int (c) for c in kalman2d.getEstimate()]
    kalman_points.append(estimated)

    # Display the trajectories and current points
    drawLines(img, kalman_points,   0,   255, 0)
    drawCross(img, estimated,       255, 255, 255)
    drawLines(img, measured_points, 255, 255, 0)
    drawCross(img, measured, 0,   0,   255)

    cv2.drawContours(frame, largest_contour, -1, (0,255,0), 3)
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('masked-frame',fgmask)
    cv2.imshow('track points - cross: red=measured, green=kalman',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
