import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from util import openFile, showImage
import colorsys
from mpl_toolkits.mplot3d import Axes3D


def openImage():
    """
        This function opens an image using OpenCV library and resizes it for faster processing.
        Parameters:
                    none
        Returns:
                 an image.
    """
    # Open Image, hardcoded for testing
    # file_path = openFile()
    file_path = "demo_beta_problems.jpeg"
    # demo_beta_problems.jpeg | demo_image1.jpg
    img = cv2.imread(file_path,1)

    # Image can be resized to a standard size to speed up processing.
    c = 1000.0/img.shape[0]
    x = int(img.shape[0] * c)
    y = int(img.shape[1] * c)
    img = cv2.resize(img, (y,x))

    return img

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny_edge_detection(image, sigma = 0.2):
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged

def overlay(img, canny, MIN_AREA = 30, MAX_AREA = 4000):
    masked = img.copy()
    _, cnts, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hulls = [cv2.convexHull(c) for c in cnts]
    for h in hulls:
            # if the contour is too small, ignore it
            area = cv2.contourArea(h)
            if area > MIN_AREA and area < MAX_AREA:
                cv2.polylines(masked, [h], True, (255,255,0), 1)
    return masked

""" Object detection """

def buildDetector(minArea = 25):
    """
    This function builds and returns a blob detector using the OpenCV library. It sets up the parameters for the detector and
    filters the blobs based on size, circularity, convexity and inertia.

        Parameters:
            minArea (int): The function takes one optional argument, minArea, which represents the minimum area for the blobs to be detected. The default value is 25.
        Returns:
            detector: The function returns the SimpleBlobDetector object that can be used to detect blobs in an image.

        Reference: https://docs.opencv.org/4.x/d0/d7a/classcv_1_1SimpleBlobDetector.html
    """

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 0
    params.maxThreshold = 255

    # Filter by Area.
    params.filterByArea = True
    params.minArea = minArea

    params.filterByColor = False

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.1
        
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.05
    params.maxInertiaRatio = 1

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else : 
        detector = cv2.SimpleBlobDetector_create(params)

    return detector


def findHolds(img, detector = None):
    """
    This function finds the holds in an image using a combination of edge detection, contour detection and blob detection.
    
        Parameters:
            img (img): It takes in an image.
            detector (cv2.SimpleBlobDetector class): An optional detector object, which should be of the cv2.SimpleBlobDetector class. If a detector is not provided, it will be built with default parameters. The default value is None.
        Returns:
            keypoints (): A list of the coordinates of the centers of the holds 
            hulls (): A list of the convex hulls of the holds.
"""
    blurred = blur(img, 5)
    grayscale = gray(blurred)
    canny = canny_edge_detection(grayscale)
    # overlay = img.copy()
    # overlay[np.where(canny)] = (255, 255, 255)

    # Finds the contours of the image, without retaining the hierarchy
    contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours_refined = []
    MIN_AREA = 25
    for contour in contours:
        area = cv2.contourArea(contour)
        print(area)
        if(area > MIN_AREA):
            contours_refined.append(contour)


    hulls = map(cv2.convexHull, contours_refined)

    # Draws contours onto a blank canvas
    mask = np.zeros(img.shape,np.uint8)
    cv2.drawContours(mask,contours_refined,-1,(255,255,255),-1)
    # cv2.drawContours(mask,hulls,-1,(255,255,255),-1)

    showImage(mask)

    if detector == None:
        # Set up the detector with default parameters.
        detector = buildDetector()

    keypoints = detector.detect(mask)
    
    return keypoints, hulls

# def is_contours_bad(keypoints):
#     print(type(keypoints))
#     average_area = cv2.mean(keypoints.size)

#     for key in enumerate(keypoints):
#         size = int(math.ceil(key.size))
#         if(size < 0.5 * average_area): 
#             keypoints.remove(key)

#     return keypoints


def findColors(img, keypoints):
    # If no keypoints return nothing
    if (keypoints == []):
        return []

    # Shift colorspace to HLS
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # Preallocate space for color array corresponding to keypoints
    colors = np.empty([len(keypoints),3])

    # Iterates through the keypoints and finds the most common
    # color at each keypoint.
    for i, key in enumerate(keypoints):
    	x = int(key.pt[0])
    	y = int(key.pt[1])

    	size = int(math.ceil(key.size)) 

        #Finds a rectangular window in which the keypoint fits
    	br = (x + size, y + size)	
    	tl = (x - size, y - size)

    	colors[i] = getColorBin(hsv,tl,br)
    
    return colors


""" Visualization """
def drawCircled(img, keypoints, contours):
    for i, key in enumerate(keypoints):
        x = int(key.pt[0])
        y = int(key.pt[1])

        size = int(math.ceil(key.size)) 

        # Draw a circle surrounding each hold instead
        cv2.circle(img,(x, y), size/2,(0,0,255),2)

    #OpenCV uses BGR format, so that'll need to be reversed for display
    img = img[...,::-1]

    # Display the resulting frame
    fig = plt.imshow(img)
    plt.title("Image with Keypoints")
    plt.show()

def drawOutlined(img, keypoints, contours):
    # Draws contours onto img
    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

    #OpenCV uses BGR format, so that'll need to be reversed for display
    img = img[...,::-1]

    # Display the resulting frame
    fig = plt.imshow(img)
    plt.title("Image with Keypoints")
    plt.show()


# def plotColors(colors):

#     # Build 3D scatterplot
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Initialize arrays
#     hs = []
#     ls = []
#     ss = []

#     # Color data is mapped between 0 and 1
#     colors = colors/256

#     for color in colors:
#         hs.append(color[0])
#         ls.append(color[1])
#         ss.append(color[2])

#     # Regain RGB Values to color each data point
#     colorsRGB = map(colorsys.hls_to_rgb,hs,ls,ss) 

#     # Plot points in HLS space
#     ax.scatter(hs, ls, ss, c=colorsRGB, marker='o')

#     ax.set_xlabel('Hue')
#     ax.set_ylabel('Lightness')
#     ax.set_zlabel('Saturation')

#     ax.set_xlim(0,1)
#     ax.set_ylim(0,1)
#     ax.set_zlim(0,1)

#     plt.title("Color Space of Keypoints")
#     plt.show()
