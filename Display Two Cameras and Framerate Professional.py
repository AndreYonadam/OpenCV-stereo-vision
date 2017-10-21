import cv2
import time

cv2.namedWindow("Left Camera") # New window for previewing the left camera

cv2.namedWindow("Right Camera") # New window for previewing the right camera

vcLeft = cv2.VideoCapture(1) # Load video campture for the left camera
#vcLeft.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,420);
#vcLeft.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,340);

vcRight = cv2.VideoCapture(2) # Load video capture for the right camera
#vcRight.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,420);
#vcRight.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,340);

firstTime = time.time() # First time log

totalFramesPassed = 0 # Number of frames passed

if vcLeft.isOpened() and vcRight.isOpened():
	rvalLeft, frameLeft = vcLeft.read()
	rvalRight, frameRight = vcRight.read()

else:
	rvalLeft = False
	rvalRight = False

while rvalLeft and rvalRight: # If the cameras are opened

	rvalLeft, frameLeft = vcLeft.read()

	rvalRight, frameRight = vcRight.read()

	cv2.putText(frameLeft, "FPS : " + str(totalFramesPassed / (time.time() - firstTime)),(40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 150, 2, 10)

	cv2.imshow("Left Camera", frameLeft)

	cv2.putText(frameRight, "FPS : " + str(totalFramesPassed / (time.time() - firstTime)),(40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 150, 2, 10)

	cv2.imshow("Right Camera", frameRight)

	key = cv2.waitKey(20)

	totalFramesPassed = totalFramesPassed + 1 # One frame passed, increment

	if key == 27:

		break

# When the user hits escape, the windows will get destroyed

cv2.destroyWindow("Left Camera")

cv2.destroyWindow("Right Camera")

vcLeft.release()

vcRight.release()