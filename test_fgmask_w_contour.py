import numpy as np
import cv2
import cv2.cv as cv

#init
cap = cv2.VideoCapture(0)
fgbg = cv2.BackgroundSubtractorMOG(30,3,0.6,20)
scale_down = 4

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (320,240))
    frame1 = frame.copy()
    # Our operations on the frame come here
    fgmask = fgbg.apply(frame)

    #test CONTOURS
    contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    largest_contour = None

    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area:
          max_area = area
          largest_contour = contour
    if not largest_contour == None:
        moment = cv2.moments(largest_contour)
        if moment["m00"] > 100 / scale_down:
          rect = cv2.minAreaRect(largest_contour)
          rect = ((rect[0][0] * scale_down, rect[0][1] * scale_down), (rect[1][0] * scale_down, rect[1][1] * scale_down), rect[2])
          box = cv2.cv.BoxPoints(rect)
          box = np.int0(box)
          cv2.drawContours(frame1,[box], 0, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('frame',frame1)
    cv2.imshow('masked-frame',fgmask)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()