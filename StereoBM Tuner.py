import cv2
import time
import numpy as np
from Tkinter import *

oldVal = 15
def oddVals(n):
        global oldVal
        n = int(n)
        if not n % 2:
                window_size.set(n+1 if n > oldVal else n-1)
                oldVal = window_size.get()

minDispValues = [16,32,48,64]
def minDispCallback(n):
        n = int(n)
        newvalue = min(minDispValues, key=lambda x:abs(x-float(n)))
        min_disp.set(newvalue)

# Display the sliders to control the stereo vision 
master = Tk()

master.title("StereoBM Settings");

min_disp = Scale(master, from_=16, to=64, command=minDispCallback, length=600, orient=HORIZONTAL, label="Minimum Disparities")
min_disp.pack()
min_disp.set(16)

window_size = Scale(master, from_=5, to=255, command=oddVals, length=600, orient=HORIZONTAL, label="Window Size")
window_size.pack()
window_size.set(15)

Disp12MaxDiff = Scale(master, from_=5, to=30, length=600, orient=HORIZONTAL, label="Max Difference")
Disp12MaxDiff.pack()
Disp12MaxDiff.set(0)

UniquenessRatio = Scale(master, from_=0, to=30, length=600, orient=HORIZONTAL, label="Uniqueness Ratio")
UniquenessRatio.pack()
UniquenessRatio.set(15)

SpeckleRange = Scale(master, from_=0, to=60, length=600, orient=HORIZONTAL, label="Speckle Range")
SpeckleRange.pack()
SpeckleRange.set(34)

SpeckleWindowSize = Scale(master, from_=60, to=150, length=600, orient=HORIZONTAL, label="Speckle Window Size")
SpeckleWindowSize.pack()
SpeckleWindowSize.set(100)

master.update()

vcLeft = cv2.VideoCapture(0) # Load video campture for the left camera
#vcLeft.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,420);
#vcLeft.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,340);

vcRight = cv2.VideoCapture(1) # Load video capture for the right camera
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

        frameLeftNew = cv2.cvtColor(frameLeft, cv2.COLOR_BGR2GRAY)

        frameRightNew = cv2.cvtColor(frameRight, cv2.COLOR_BGR2GRAY)

        num_disp = 112 - min_disp.get()

        stereo = cv2.StereoBM_create(numDisparities = num_disp, blockSize = window_size.get())

        stereo.setMinDisparity(min_disp.get())

        stereo.setNumDisparities(num_disp)

        stereo.setBlockSize(window_size.get())

        stereo.setDisp12MaxDiff(Disp12MaxDiff.get())

        stereo.setUniquenessRatio(UniquenessRatio.get())

        stereo.setSpeckleRange(SpeckleRange.get())

        stereo.setSpeckleWindowSize(SpeckleWindowSize.get())

        disparity = stereo.compute(frameLeftNew, frameRightNew).astype(np.float32) / 16.0

        disp_map = (disparity - min_disp.get())/num_disp

        cv2.imshow("Disparity", disp_map)

        master.update() # Update the slider options

        key = cv2.waitKey(20)

        totalFramesPassed = totalFramesPassed + 1 # One frame passed, increment

        if key == 27:

                break


vcLeft.release()

vcRight.release()
