# coding=utf-8
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def explain_keypoint(kp):
    print('angle\n', kp.angle)
    print('\nclass_id\n', kp.class_id)
    print('\noctave (image scale where feature is strongest)\n', kp.octave)
    print('\npt (x,y)\n', kp.pt)
    print('\nresponse\n', kp.response)
    print('\nsize\n', kp.size)


def showing_sift_features(img1, img2, key_points):
    return plt.imshow(cv2.drawKeypoints(img1, key_points, img2.copy()))


print("######################### DEBUT PREPARATION DES IMAGES SUR LESQUELS ON SOUHAITE FAIRE LE TEST ##################################")
Image1 = cv2.imread("face.jpg")
Image2 = cv2.imread("original.jpg")

Image1_gray = cv2.cvtColor(Image1, cv2.COLOR_BGR2GRAY)  # On grise les images
Image2_gray = cv2.cvtColor(Image2, cv2.COLOR_BGR2GRAY)

sift_initialize = cv2.xfeatures2d.SIFT_create()

Image1_key_points, Image1_descriptors = sift_initialize.detectAndCompute(Image1_gray, None)
Image2_key_points, Image2_descriptors = sift_initialize.detectAndCompute(Image2_gray, None)


print('this is an example of a single SIFT keypoint:\n* * *')
explain_keypoint(Image1_key_points[0])

print("######################### CREATION DES POINTS POUR LA TRANSFORMATION AFFINE ##################################")

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(Image1_descriptors, Image2_descriptors, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.3 * n.distance:
        good.append(m)
        #print("m", m)
        #print("n", n)

print("Resultat du test de distance, on regroupe les couple de coordonnées de points photo a et photo b ensemble dans good :\n")
print("good = ", good)

# Featured matched keypoints from images 1 and 2
pts1 = np.float32([Image1_key_points[m.queryIdx].pt for m in good])
pts2 = np.float32([Image2_key_points[m.trainIdx].pt for m in good])

print("On peut récuperer les points [x y] de seulement la premiere photo :\n", pts1)
"""print("points arranges 2", pts2)"""

print("On peut récuperer la coordonnée x du premier point enregistré de la premiere photo", pts1[0][0])

# On dessine chaque point d'interet sur lequel on travail sur les photos
cv2.circle(Image1, (pts1[0][0], pts1[0][1]), 10, (255, 0, 0), -1)
cv2.circle(Image2, (pts2[0][0], pts2[0][1]), 10, (255, 0, 0), -1)

cv2.circle(Image1, (pts1[6][0], pts1[6][1]), 10, (255, 0, 0), -1)
cv2.circle(Image2, (pts2[6][0], pts2[6][1]), 10, (255, 0, 0), -1)

cv2.circle(Image1, (pts1[2][0], pts1[2][1]), 10, (255, 0, 0), -1)
cv2.circle(Image2, (pts2[2][0], pts2[2][1]), 10, (255, 0, 0), -1)

# On selectionne 3 points et on parse l'ensemble des points dans le bon format
pt1 = np.float32([pts1[0], pts1[6], pts1[2]])
pt2 = np.float32([pts2[0], pts2[6], pts2[2]])

# On donne les points à la mthde qui nous permet d'obtenir la matrice de Transformation affine
matrix = cv2.getAffineTransform(pt1, pt2)
print("Matrice de transformation affine \n", matrix)

#remove of the scaling of the matrix

print("\n")
print("A = ", matrix[0][0])
print("B = ", matrix[0][1])
print("D = ", matrix[1][0])
print("E = ", matrix[1][1])

# div A & B by sqrt(A*A+B*B)
a = matrix[0][0]/((matrix[0][0]*matrix[0][0]+matrix[0][1]*matrix[0][1])**0.5)
b = matrix[0][1]/((matrix[0][0]*matrix[0][0]+matrix[0][1]*matrix[0][1])**0.5)

# div D and E by sqrt(D*D+E*E)
d = matrix[1][0]/((matrix[1][0]*matrix[1][0]+matrix[1][1]*matrix[1][1])**0.5)
e = matrix[1][1]/((matrix[1][0]*matrix[1][0]+matrix[1][1]*matrix[1][1])**0.5)

cos1_Angle = np.degrees(np.arccos(a))
sin1_Angle = - np.degrees(np.arcsin(b))
sin2_Angle = np.degrees(np.arcsin(d))
cos2_Angle = np.degrees(np.arccos(e))

print("\n")
print("Angle 1 = ", cos1_Angle)
print("Angle 2 = ", sin1_Angle)
print("Angle 3 = ", sin2_Angle)
print("Angle 4 = ", cos2_Angle)


cv2.waitKey(0)
cv2.destroyAllWindows()

