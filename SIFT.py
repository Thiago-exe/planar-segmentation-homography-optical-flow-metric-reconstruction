import cv2
import numpy as np

# Load images
img1 = cv2.imread(r'WhatsApp Image 2023-03-16 at 21.44.00.jpeg')
img2 = cv2.imread(r'WhatsApp Image 2023-03-16 at 21.44.00 (1).jpeg')

# Convert images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

gaussian_pyramid = []
gaussian_pyramid2 = []

# define number of octaves and levels
num_octaves = 4
num_levels = 5

# apply Gaussian blur for each octave and level
for i in range(num_octaves):
    octave = []
    for j in range(num_levels):
        # apply Gaussian blur with increasing kernel size for each level
        ksize = (2*j+1, 2*j+1)
        blurred = cv2.GaussianBlur(gray1, ksize, 0)
        octave.append(blurred)
    gaussian_pyramid.append(octave)
    # downsample the image by half for the next octave
    gray1 = blurred


for i in range(num_octaves):
    octave = []
    for j in range(num_levels):
        # apply Gaussian blur with increasing kernel size for each level
        ksize = (2*j+1, 2*j+1)
        blurred = cv2.GaussianBlur(gray2, ksize, 0)
        octave.append(blurred)
    gaussian_pyramid2.append(octave)
    # downsample the image by half for the next octave
    gray2 = blurred

# display the Gaussian pyramid
cv2.imshow('gaussian 2', gray2)
cv2.imshow('gaussian 1', gray1)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

img1 = cv2.drawKeypoints(gray1,kp1,img1)
cv2.imshow('keypoints', img1)


# Display image

cv2.waitKey(0)
cv2.destroyAllWindows()
