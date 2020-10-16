# coding=utf-8
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def showing_sift_features(img1, img2, key_points):
    return plt.imshow(cv2.drawKeypoints(img1, key_points, img2.copy()))

# Importation des images
Image1 = cv2.imread("face.jpg")
Image2 = cv2.imread("original.jpg")

# Conversion des images en gris. SIFT a besoin d'images grises pour effectuer ses opérations.
Image1_gray = cv2.cvtColor(Image1, cv2.COLOR_BGR2GRAY)
Image2_gray = cv2.cvtColor(Image2, cv2.COLOR_BGR2GRAY)

# Chargement de l'algorithme
sift_initialize = cv2.xfeatures2d.SIFT_create()

# Détection des points d'intérêts avec la méthode detectAndCompute().
Image1_key_points, Image1_descriptors = sift_initialize.detectAndCompute(Image1_gray, None)
Image2_key_points, Image2_descriptors = sift_initialize.detectAndCompute(Image2_gray, None)

showing_sift_features(Image1_gray, Image1, Image1_key_points);

# Utilisation de NORM_L2 pour calculer la distance de Manhattan entre les deux points,
# qui sera utilisée pour effectuer le matching.
norm = cv2.NORM_L2
bruteForce = cv2.BFMatcher(norm)

# Application de bruteForce.match pour faire la correspondance entre les déscripteurs.
# Les correspondances sont ensuite triées en fonction de la distance de Manhattan.
matches = bruteForce.match(Image1_descriptors, Image2_descriptors)
matches = bruteForce.match(Image1_descriptors, Image2_descriptors)
matches = sorted(matches, key=lambda match: match.distance)

# Dessin des correspondances.
matched_img = cv2.drawMatches(Image1, Image1_key_points, Image2, Image2_key_points, matches[:100], Image2.copy())

# resize image
scale_percent = 20  # percent of original size
width = int(matched_img.shape[1] * scale_percent / 100)
height = int(matched_img.shape[0] * scale_percent / 100)
dim = (width, height)
sift = cv2.resize(matched_img, dim, interpolation=cv2.INTER_AREA)

# Affichage
cv2.imshow("Feature matching with SIFT", sift)

print(Image1_descriptors)
print(Image2_descriptors)
print(Image1_key_points)
print(Image2_key_points)


cv2.waitKey(0)
cv2.destroyAllWindows()
