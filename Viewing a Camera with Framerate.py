import cv2
import time

cv2.namedWindow("My camera") # New window for previewing the left camera
vcLeft = cv2.VideoCapture(0) # Load video campture for the left camera
vcLeft.set(3,640)
vcLeft.set(4,480)

# Declare variables to hold the average
currentSum = 0.0
numberOfTimeSamples = 0.0
previousTime = int(round(time.time() * 1000)) # Previous time in milliseconds

if vcLeft.isOpened():
	rvalLeft, frameLeft = vcLeft.read()
else:
	rvalLeft = False

while rvalLeft: # If the cameras are opened
	cv2.imshow("My camera", frameLeft)
	rvalLeft, frameLeft = vcLeft.read()
	
	currentSum = currentSum + 1000/(int(round(time.time() * 1000)) - previousTime)
	numberOfTimeSamples = numberOfTimeSamples + 1.0
	cv2.putText(frameLeft,"Average FPS: " + str( currentSum/numberOfTimeSamples), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 150, 2, 10)

	previousTime = int(round(time.time() * 1000)) # Set current time in milliseconds
	key = cv2.waitKey(20)
	if key == 27:
		break

# When the user hits escape, the windows will get destroyed
cv2.destroyWindow("Left Camera")
vcLeft.release()