import numpy as np
import cv2
import cv2.cv as cv

#init
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
fgbg = cv2.BackgroundSubtractorMOG(30,3,0.6,20)
prev = frame
step = 32

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (320,240))
    vis = frame
    frame1 = frame.copy()
    # Our operations on the frame come here
    fgmask = fgbg.apply(frame)

    #test CONTOURS
    ret,thresh = cv2.threshold(fgmask,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        cv2.drawContours(frame1, contours, -1, (0,255,0), 3)

        flow = cv2.calcOpticalFlowFarneback(prev, fgmask, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = frame.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
        fx, fy = flow[y,x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = frame
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

    prev = fgmask

    # Display the resulting frame
    cv2.imshow('flow', vis)
    cv2.imshow('frame',frame1)
    cv2.imshow('masked-frame',fgmask)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()