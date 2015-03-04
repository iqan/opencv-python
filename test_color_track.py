# obj tracking by color -- iqan & rohil 03 March,2015

import cv2, math
import numpy as np

message = '''
---------------------------------------
|Color Tracker                        |
---------------------------------------
|Keys:                                |
| 1 - Track Blue Color                |
| 2 - Track Green Color               |
| 3 - Track Red Color                 |
---------------------------------------
|                             made by |
|                        Iqan & Rohil |
---------------------------------------
'''

print message

#initialization
cv2.namedWindow("ColourTrackerWindow", cv2.CV_WINDOW_AUTOSIZE)
capture = cv2.VideoCapture(0)
scale_down = 4
#init red tracker
lower = np.array([0, 150, 0],np.uint8)
upper = np.array([5, 255, 255],np.uint8)

while True:
  f, orig_img = capture.read()

  orig_img = cv2.flip(orig_img, 1)
  img = cv2.GaussianBlur(orig_img, (5,5), 0)
  img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
  img = cv2.resize(img, (len(orig_img[0]) / scale_down, len(orig_img) / scale_down))

  #red range
  red_lower = np.array([0, 150, 0],np.uint8)
  red_upper = np.array([5, 255, 255],np.uint8)
  #green range
  green_lower = np.array([50, 100, 100])
  green_upper = np.array([70, 255, 255])
  #blue range
  blue_lower = np.array([110,50,50])
  blue_upper = np.array([130,255,255])  

  #masking
  binary = cv2.inRange(img, lower, upper)
  dilation = np.ones((15, 15), "uint8")
  binary = cv2.dilate(binary, dilation)

  #finding contours
  contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  max_area = 0
  largest_contour = None

  for idx, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area > max_area:
      max_area = area
      largest_contour = contour
  if not largest_contour == None:
    moment = cv2.moments(largest_contour)
    if moment["m00"] > 1000 / scale_down:
      rect = cv2.minAreaRect(largest_contour)
      rect = ((rect[0][0] * scale_down, rect[0][1] * scale_down), (rect[1][0] * scale_down, rect[1][1] * scale_down), rect[2])
      box = cv2.cv.BoxPoints(rect)
      box = np.int0(box)
      cv2.drawContours(orig_img,[box], 0, (0, 0, 255), 2)
      cv2.imshow("ColourTrackerWindow", orig_img)
      
      ch = 0xFF & cv2.waitKey(5)
      if ch == 27:
          break
      if ch == ord('1'):
          lower = blue_lower
          upper = blue_upper
          print 'Tracking Blue'
      if ch == ord('2'):
          lower = green_lower
          upper = green_upper
          print 'Tracking Green'
      if ch == ord('3'):
          lower = red_lower
          upper = red_upper
          print 'Tracking Red'
cv2.destroyWindow("ColourTrackerWindow")
capture.release()

# note for Iqan
# select range for color > mask image >
#  `--> find contour > largest_contour > 
#  `--> find moments (here area) > max_area 
#  `--> max area is greater than thresh > Track !! BINGO !!