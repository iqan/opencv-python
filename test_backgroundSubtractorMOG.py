#background subtractor
# note : use with constant luminance and clear background

import numpy as np
import cv2
import cv2.cv as cv
#import matplotlib.pyplot as plt

#init
cap = cv2.VideoCapture(0)
fgbg = cv2.BackgroundSubtractorMOG(30,3,0.6,20)
#plt.figure()
#plt.hold(True)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    fgmask = fgbg.apply(frame)

    #test CONTOURS
    ret,thresh = cv2.threshold(fgmask,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
   # if len(contours) > 0:
    #    m = np.mean(contours[0],axis=0)
     #   #print 'm[0,0] = ',m[0,0],'||  m[0,1] = ',m[0,1]
      #  plt.plot(m[0,0],m[0,1],'ob')

    cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('masked-frame',fgmask)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
