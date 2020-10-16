import cv2
import numpy as np

img1 = cv2.imread("face.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("original.jpg", cv2.IMREAD_GRAYSCALE)

# ORB Detector
orb = cv2.ORB_create(nfeatures=1500)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
# Brute Force Matching ORB
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

# SURF Detector
surf = cv2.xfeatures2d.SURF_create()
keypoints_surf1, descriptors_surf1 = surf.detectAndCompute(img1, None)
keypoints_surf2, descriptors_surf2 = surf.detectAndCompute(img2, None)
# Brute Force Matching SURF
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
matching_result_surf = cv2.drawMatches(img1, keypoints_surf1, img2, keypoints_surf2, matches[:50], None, flags=2)

# resize image
scale_percent = 20
width = int(matching_result.shape[1] * scale_percent / 100)
height = int(matching_result.shape[0] * scale_percent / 100)
dim = (width, height)
orb = cv2.resize(matching_result, dim, interpolation=cv2.INTER_AREA)
surf = cv2.resize(matching_result_surf, dim, interpolation=cv2.INTER_AREA)

#cv2.imshow("Img1", img1)
#cv2.imshow("Img2", img2)

cv2.imshow("Matching result with ORB", orb)
cv2.imshow("Matching result with SURF", surf)

cv2.waitKey(0)
cv2.destroyAllWindows()
