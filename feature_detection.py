# coding=utf-8
import cv2
import numpy as np

img = cv2.imread("face.jpg", cv2.IMREAD_GRAYSCALE)

# Chargement des trois algorithmes
sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create(nfeatures=1500)

# Detection des points d'intérêts et des descripteurs de chaque algorithme.
# Un point d'interets est la position ou le feature a été détecté, tandis que
# le descripteur est un tableau contenant des chiffres pour décrire ce feature.
# Lorsque les descripteurs sont similaires, cela signifie que le feature est également similaire.

keypoints_sift, descriptors = sift.detectAndCompute(img, None)
keypoints_surf, descriptors = surf.detectAndCompute(img, None)
keypoints_orb, descriptors = orb.detectAndCompute(img, None)


# print(keypoints_sift[0])
# print(descriptors)

# Affichage des caractéristiques d'un point d'intérêts
def explain_keypoint(kp):
    print 'angle\n', kp.angle
    print '\nclass_id\n', kp.class_id
    print '\noctave (image scale where feature is strongest)\n', kp.octave
    print '\npt (x,y)\n', kp.pt
    print '\nresponse\n', kp.response
    print '\nsize\n', kp.size


print 'this is an example of a single SIFT keypoint:\n* * *'
explain_keypoint(keypoints_sift[0])

# Dessin des points d'intérêts sur l'image
img = cv2.drawKeypoints(img, keypoints_sift, None)
img2 = cv2.drawKeypoints(img, keypoints_surf, None)
img3 = cv2.drawKeypoints(img, keypoints_orb, None)

# resize image
scale_percent = 50  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
imgSIFT = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
imgSURF = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
imgORB = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# Affichage des résultats
cv2.imshow("Image SIFT", imgSIFT)
cv2.imshow("Image SURF", imgSURF)
cv2.imshow("Image ORB", imgORB)

cv2.waitKey(0)
cv2.destroyAllWindows()
