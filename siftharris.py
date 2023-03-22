import cv2
import numpy as np

# Load images
img1 = cv2.imread(r'WhatsApp Image 2023-03-16 at 21.44.00.jpeg')
img2 = cv2.imread(r'WhatsApp Image 2023-03-16 at 21.44.00 (1).jpeg')

# Convert images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Find corners using Harris corner detector
corners = cv2.cornerHarris(gray1, blockSize=2, ksize=3, k=0.04)
corners = cv2.dilate(corners, None)
threshold = 0.01 * corners.max()
corner_img = np.copy(img1)
corner_img[corners > threshold] = [0, 0, 255]

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Draw keypoints on image
kp_img1 = np.copy(img1)
kp_img2 = np.copy(img2)
cv2.drawKeypoints(img1, kp1, kp_img1)
cv2.drawKeypoints(img2, kp2, kp_img2)

# Display images
cv2.imshow('Corners', corner_img)
cv2.imshow('Keypoints 1', kp_img1)
cv2.imshow('Keypoints 2', kp_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
