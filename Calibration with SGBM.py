# Imports
import cv2
import numpy as np

# Constants
leftCameraNumber = 2 # Number for left camera
rightCameraNumber = 1 # Number for right camera

numberOfChessRows = 6
numberOfChessColumns = 8
chessSquareSize = 30 # Length of square in millimeters

numberOfChessColumns = numberOfChessColumns - 1 # Update to reflect how many corners are inside the chess board
numberOfChessRows = numberOfChessRows - 1

objp = np.zeros((numberOfChessColumns*numberOfChessRows,3), np.float32)
objp[:,:2] = np.mgrid[0:numberOfChessRows,0:numberOfChessColumns].T.reshape(-1,2)*chessSquareSize

objectPoints = []
leftImagePoints = []
rightImagePoints = []

parameterCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Code
print("Press \"n\" when you're done caputing checkerboards.")

vcLeft = cv2.VideoCapture(leftCameraNumber) # Load video campture for the left camera
vcLeft.set(cv2.CAP_PROP_FRAME_WIDTH,640*3/2);
vcLeft.set(cv2.CAP_PROP_FRAME_HEIGHT,480*3/2);

vcRight = cv2.VideoCapture(rightCameraNumber) # Load video capture for the right camera
vcRight.set(cv2.CAP_PROP_FRAME_WIDTH,640*3/2);
vcRight.set(cv2.CAP_PROP_FRAME_HEIGHT,480*3/2);

if vcLeft.isOpened() and vcRight.isOpened():
	rvalLeft, frameLeft = vcLeft.read()
	rvalRight, frameRight = vcRight.read()

else:
	rvalLeft = False
	rvalRight = False
	
# Number of succesful recognitions
checkerboardRecognitions = 0

while rvalLeft and rvalRight: # If the cameras are opened

	vcLeft.grab();
	
	vcRight.grab();

	rvalLeft, frameLeft = vcLeft.retrieve()

	rvalRight, frameRight = vcRight.retrieve()
	
	frameLeftNew = cv2.cvtColor(frameLeft, cv2.COLOR_BGR2GRAY)

	frameRightNew = cv2.cvtColor(frameRight, cv2.COLOR_BGR2GRAY)
	
	foundPatternLeft, cornersLeft = cv2.findChessboardCorners(frameLeftNew, (numberOfChessRows, numberOfChessColumns), None, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
	
	foundPatternRight, cornersRight = cv2.findChessboardCorners(frameRightNew, (numberOfChessRows, numberOfChessColumns), None, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
	
	
	if foundPatternLeft and foundPatternRight: # If found corners in this frame
		
		# Process the images and display the count of checkboards in our array
		checkerboardRecognitions = checkerboardRecognitions + 1
		print("Checker board recognitions: " + str(checkerboardRecognitions))
		
		objectPoints.append(objp)
		
		exactCornersLeft = cv2.cornerSubPix(frameLeftNew, cornersLeft, (11, 11), (-1, -1), parameterCriteria);
		leftImagePoints.append(exactCornersLeft)
		
		exactCornersRight = cv2.cornerSubPix(frameRightNew, cornersRight, (11, 11), (-1, -1), parameterCriteria);
		rightImagePoints.append(exactCornersRight)

		frameLeft = cv2.drawChessboardCorners(frameLeft, (numberOfChessRows, numberOfChessColumns), (exactCornersLeft), True);
		
		frameRight = cv2.drawChessboardCorners(frameRight, (numberOfChessRows, numberOfChessColumns), (exactCornersRight), True);
		
		
	# Display current webcams regardless if board was found or not
	cv2.imshow("Left Camera", frameLeft)

	cv2.imshow("Right Camera", frameRight)
		

	key = cv2.waitKey(250) # Give the frame some time
	
	if key == ord('n'):

		break
				
cameraMatrixLeft = np.zeros( (3,3) )
cameraMatrixRight = np.zeros( (3,3) )
distortionLeft = np.zeros( (8,1) )
distortionRight = np.zeros( (8,1) )
height, width = frameLeft.shape[:2]

rms, leftMatrix, leftDistortion, rightMatrix, rightDistortion, R, T, E, F = cv2.stereoCalibrate(objectPoints, leftImagePoints, rightImagePoints,  cameraMatrixLeft, distortionLeft, cameraMatrixRight, distortionRight, (width, height),parameterCriteria, flags=0)

arr1 = np.arange(8).reshape(2, 4)
arr2 = np.arange(10).reshape(2, 5)
np.savez('camera_calibration.npz', leftMatrix=leftMatrix, leftDistortion=leftDistortion, rightMatrix=rightMatrix, rightDistortion=rightDistortion, R=R, T=T, E=E, F=F)
print("Calibration Settings Saved to File!")

print("RMS:")
print(rms)
print("Left Matrix:")
print(leftMatrix)
print("Left Distortion:")
print(leftDistortion)
print("Right Matrix:")
print(rightMatrix)
print("Right Distortion:")
print(rightDistortion)
print("R:")
print(R)
print("T:")
print(T)
print("E:")
print(E)
print("F:")
print(F)


leftRectTransform, rightRectTransform, leftProjMatrix, rightProjMatrix, _, _, _ = cv2.stereoRectify(leftMatrix, leftDistortion, rightMatrix, rightDistortion,  (width, height), R, T, alpha=-1);
leftMapX, leftMapY = cv2.initUndistortRectifyMap(leftMatrix, leftDistortion, leftRectTransform, leftProjMatrix, (width, height), cv2.CV_32FC1);
rightMapX, rightMapY = cv2.initUndistortRectifyMap(rightMatrix, rightDistortion, rightRectTransform, rightProjMatrix, (width, height), cv2.CV_32FC1);

minimumDisparities = 0
maximumDisparities = 128

stereo = cv2.StereoSGBM_create(minimumDisparities, maximumDisparities, 18)

while True: # If the cameras are opened
	vcLeft.grab();
		
	vcRight.grab();
	
	rvalLeft, frameLeft = vcLeft.retrieve()

	rvalRight, frameRight = vcRight.retrieve()
	
	frameLeftNew = cv2.cvtColor(frameLeft, cv2.COLOR_BGR2GRAY)

	frameRightNew = cv2.cvtColor(frameRight, cv2.COLOR_BGR2GRAY)

	leftRectified = cv2.remap(frameLeftNew, leftMapX, leftMapY, cv2.INTER_LINEAR);
	
	rightRectified = cv2.remap(frameRightNew, rightMapX, rightMapY, cv2.INTER_LINEAR);

	disparity = stereo.compute(leftRectified, rightRectified)
	
	cv2.filterSpeckles(disparity, 0, 6000, maximumDisparities);

	cv2.imshow("Normalized Disparity", (disparity/16.0 - minimumDisparities)/maximumDisparities);
	
	cv2.imshow("Left Camera", leftRectified)

	cv2.imshow("Right Camera", rightRectified)
		

	key = cv2.waitKey(10) # Give the frame some time

	if key == 27:

		break

print("Finished!")