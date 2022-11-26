# Video capture example

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from holdDetector2 import findHolds, findColors


# Starting path
path = 'VideoTestTrimmed.mp4'
# path = 'C:\Users\James\Downloads\warwick\cs407\neuralclimb\NeuralClimb\Hold Detection\Test3.gif'
# C:\Users\James\Downloads\warwick\cs407 group project\neuralclimb\NeuralClimb\Hold Detection
# Open capture
cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
print(cap.isOpened())

# Find dimensions of capture for output video
ret, frame = cap.read()
y,x,ch = frame.shape
outShape = (0,0)
if x > y:
    axis = 0
    y *= 2
else:
    axis = 1
    x *= 2
shape = (x,y)

# Open VideoWriter object
#codec = cv2.cv.CV_FOURCC('Y','V','1','2')
out = cv2.VideoWriter(path[:-4] + '-out.avi',-1, 30.0, shape,True)

total_frames = 0
total_holds = 0


while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Find keypoints
    keypoints, hulls = findHolds(frame)
    print("Number of holds found: ", len(keypoints))


    frameWithKeypoints = cv2.drawKeypoints(frame,keypoints,-1,[0,0,255])
    #cv2.drawContours(frame,hulls,-1,[255,0,0])
    results = np.concatenate((frame, frameWithKeypoints), axis=axis)

    # Write image to video out
    out.write(results)  

    cv2.imshow('frame', results)

    total_frames = total_frames + 1
    total_holds = total_holds + len(keypoints)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

average = total_holds / total_frames
print("Average number of holds found: ", average)


# When everything done, release the capture
out.release()
cap.release()
cv2.destroyAllWindows()
