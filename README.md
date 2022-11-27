# Climbing Holds - Detection and Classification

*CS407 - The Beta Creators*

## Setup (Windows)

1. Make sure you first have a graphical application/display installed and running before running the program - I use XLaunch :)

## Running the program

# Image file
1. With XLaunch running, run `python Example.py` a prompt window should appear to select an image of choice - Look into `NeuralClimb/Hold_Detection/test_images` and select one. 
	(`compact.jpg` was kindly provided by @MPawsey and was the main image for testing!)
2. A thresholded image in black and white appears.
3. Click any key to load contours onto the original image.
4. Admire it :)

# Video File
1. As above with XLaunching, run `python Video.py` which runs `VideoTestTrimmed.mp4` in `NeuralClimb/Hold_Classification`.
2. The first frame of the video appears with holds detected.
3. Hold any key to load contours onto the video as it continues playing.

# Running it LIVE (to be completed)

## Notes

- `Example.py` runs `holdDetector.py`
- Some experimentation was done to narrow down the parameters for best detection (e.g. modifying Gaussian kernel size) - 
	`Example2.py` runs `holdDetector2.py` which takes another approach to Canny edge detection (takes into account median of image).

- Finds most holds, but not all (heavily influenced by quality and resolution of provided image)
- **CLASSIFICATION** to be completed in `NeuralClimb/Hold_Classification`

- If you have any more questions, pls msg me @tames_jam x

