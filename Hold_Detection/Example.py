"""Example2"""

# Uses holdDetector.py
# Performs 1. Gaussian Blur 2. Converts to Grayscale 3. Canny edge detection (using cv2.threshold)
# 4. Contour detection 5. Filters on contours based on area (removing ones too small) 6. Draw

import holdDetector as hd

#Open dialog to select image
img = hd.openImage()

# Set initial detector parameters
hd.buildDetector(minArea = 500)

# Finds each hold. Returns keypoints for each hold and the points that define the contours of each hold
holds, contours = hd.findHolds(img)

# Detail for each keypoint hold found
print("Number of keypoints: ", len(holds))
for i in range(0, len(holds)):
	print("Keypoint ", i, " | x coordinates: ", holds[i].pt[0], " | y coordinates: ", holds[i].pt[1])

# updated_holds = hd.is_contours_bad(holds)
hd.drawOutlined(img, holds, contours)

