import cv2
import time

vcLeft = cv2.VideoCapture(0) # Load video campture for the left camera
#vcLeft.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,420);
#vcLeft.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,340);

vcRight = cv2.VideoCapture(0) # Load video capture for the right camera
#vcRight.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,420);
#vcRight.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,340);

fileIndex = 0

if vcLeft.isOpened() and vcRight.isOpened():
	rvalLeft, frameLeft = vcLeft.read()
	rvalRight, frameRight = vcRight.read()

else:
	rvalLeft = False
	rvalRight = False

while rvalLeft and rvalRight: # If the cameras are opened

	rvalLeft, frameLeft = vcLeft.read()

	rvalRight, frameRight = vcRight.read()

	cv2.imwrite("images/LEFT/test" + str(fileIndex) + ".jpg",frameLeft)
	
	cv2.imwrite("images/RIGHT/test" + str(fileIndex) + ".jpg",frameRight)
	
	fileIndex = fileIndex + 1
	
	time.sleep(3)
	
	key = cv2.waitKey(20)

	if key == 27:

		break

# When the user hits escape, the windows will get destroyed

vcLeft.release()

vcRight.release()