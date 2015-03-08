#mod in opt flow by iqan

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
step=32
ret, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

while True:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 3, 15, 3, 5, 1.2, 0)

	h, w = img.shape[:2]
	y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
	fx, fy = flow[y,x].T
	lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
	lines = np.int32(lines + 0.5)
	vis = img
	cv2.polylines(vis, lines, 0, (0, 255, 0))
	for (x1, y1), (x2, y2) in lines:
	    cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

	prevgray = gray
	cv2.imshow('flow', vis)

	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()