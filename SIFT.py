import cv2
import numpy as np

# Load images
img1 = cv2.imread(r'D:/Python/Nova pasta/WhatsApp Image 2023-03-16 at 21.44.00 (1).jpeg')
img2 = cv2.imread(r'D:/Python/Nova pasta/WhatsApp Image 2023-03-16 at 21.44.00.jpeg')

# Convert images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Initialize BFMatcher
bf = cv2.FlannBasedMatcher()

# Match descriptors
matches = bf.match(des1, des2)

# Sort matches in order of distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw top 10 matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


# Display image
img_matches_resized = cv2.resize(img_matches, (800, 600))
cv2.imshow('Matches', img_matches_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
