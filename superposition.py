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


def overlay_transparent(large_image, small_image, x, y):
    background_width = large_image.shape[1]
    background_height = large_image.shape[0]

    if x >= background_width or y >= background_height:
        return large_image

    h, w = small_image.shape[0], small_image.shape[1]

    if x + w > background_width:
        w = background_width - x
        small_image = small_image[:, :w]

    if y + h > background_height:
        h = background_height - y
        small_image = small_image[:h]

    if small_image.shape[2] < 4:
        small_image = np.concatenate(
            [
                small_image,
                np.ones((small_image.shape[0], small_image.shape[1], 1), dtype=small_image.dtype) * 255
            ],
            axis=2,
        )

    overlay_image = small_image[..., :3]
    mask = small_image[..., 3:] / 255.0

    large_image[y:y + h, x:x + w] = (1.0 - mask) * large_image[y:y + h, x:x + w] + mask * overlay_image

    return large_image


def test_overlay(back, fore, x, y):
    rows, cols, channels = fore.shape
    trans_indices = fore[..., :3] != 0  # Where not transparent
    overlay_copy = back[y:y + rows, x:x + cols]
    overlay_copy[trans_indices] = fore[trans_indices]
    back[y:y + rows, x:x + cols] = overlay_copy

    return back


######################### PREPARATION DES IMAGES SUR LESQUELS ON SOUHAITE FAIRE LE TEST ##################################

Image1 = cv2.imread("face.jpg")
Image2 = cv2.imread("original.jpg")

Image1_gray = cv2.cvtColor(Image1, cv2.COLOR_BGR2GRAY)  # On grise les images
Image2_gray = cv2.cvtColor(Image2, cv2.COLOR_BGR2GRAY)

sift_initialize = cv2.xfeatures2d.SIFT_create()

Image1_key_points, Image1_descriptors = sift_initialize.detectAndCompute(Image1_gray, None)
Image2_key_points, Image2_descriptors = sift_initialize.detectAndCompute(Image2_gray, None)

print('this is an example of a single SIFT keypoint:\n* * *')
explain_keypoint(Image1_key_points[0])

######################### CREATION DES POINTS POUR LA TRANSFORMATION AFFINE ##################################"

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(Image1_descriptors, Image2_descriptors, k=2)

# ratio test, on garde les points d'interet les plus interessants
good = []
for m, n in matches:
    if m.distance < 0.3 * n.distance:
        good.append(m)
        # print("m", m)
        # print("n", n)

print("\nOn garde les points d'interet les plus interessants dans good = ", good)

# Featured matched keypoints from images 1 and 2
pts1 = np.float32([Image1_key_points[m.queryIdx].pt for m in good])
pts2 = np.float32([Image2_key_points[m.trainIdx].pt for m in good])

print("\n On peut récuperer les points [x y] de seulement la premiere photo :\n", pts1)
print("\n On peut récuperer la coordonnée x du premier point enregistré de la premiere photo", pts1[0][0])

# On dessine chaque point d'interet sur lequel on travail sur les photos
cv2.circle(Image1, (pts1[0][0], pts1[0][1]), 5, (255, 0, 0), -1)
cv2.circle(Image2, (pts2[0][0], pts2[0][1]), 5, (255, 0, 0), -1)

cv2.circle(Image1, (pts1[6][0], pts1[6][1]), 5, (255, 0, 0), -1)
cv2.circle(Image2, (pts2[6][0], pts2[6][1]), 5, (255, 0, 0), -1)

cv2.circle(Image1, (pts1[2][0], pts1[2][1]), 5, (255, 0, 0), -1)
cv2.circle(Image2, (pts2[2][0], pts2[2][1]), 5, (255, 0, 0), -1)

# On selectionne 3 points et on parse l'ensemble des points dans le bon format
pt1 = np.float32([pts1[0], pts1[6], pts1[2]])
pt2 = np.float32([pts2[0], pts2[6], pts2[2]])

# On donne les points à la mthde qui nous permet d'obtenir la matrice de Transformation affine
matrix = cv2.getAffineTransform(pt1, pt2)
print("\n Matrice de transformation affine \n", matrix)

print("\n Valeurs des paramètres de la matrice de transformation \n")
print("A = ", matrix[0][0])
print("B = ", matrix[0][1])
print("D = ", matrix[1][0])
print("E = ", matrix[1][1])

# div A & B by sqrt(A*A+B*B) pour enlever l'echelle
a = matrix[0][0] / ((matrix[0][0] * matrix[0][0] + matrix[0][1] * matrix[0][1]) ** 0.5)
b = matrix[0][1] / ((matrix[0][0] * matrix[0][0] + matrix[0][1] * matrix[0][1]) ** 0.5)

# div D and E by sqrt(D*D+E*E) pour enlever l'echelle
d = matrix[1][0] / ((matrix[1][0] * matrix[1][0] + matrix[1][1] * matrix[1][1]) ** 0.5)
e = matrix[1][1] / ((matrix[1][0] * matrix[1][0] + matrix[1][1] * matrix[1][1]) ** 0.5)

print("\n Valeurs des paramètres de la matrice sans l'echelle \n")
print("a = ", a)
print("b = ", b)
print("d = ", d)
print("e = ", e)

cos1_Angle = np.degrees(np.arccos(a))
sin1_Angle = - np.degrees(np.arcsin(b))
sin2_Angle = np.degrees(np.arcsin(d))
cos2_Angle = np.degrees(np.arccos(e))

print("\n Angle déduit des valeurs précédantes: les 4 nombres suivants sont censé être les mêmes \n")
print("Angle1 = ", cos1_Angle)
print("Angle2 = ", sin1_Angle)
print("Angle3 = ", sin2_Angle)
print("Angle4 = ", cos2_Angle)

##################### Début traitement des points d'intêrets ############################
showing_sift_features(Image1_gray, Image1, Image1_key_points)

norm = cv2.NORM_L2
bruteForce = cv2.BFMatcher(norm)

matches = bruteForce.match(Image1_descriptors, Image2_descriptors)
matches = bruteForce.match(Image1_descriptors, Image2_descriptors)
matches = sorted(matches, key=lambda match: match.distance)

matched_img = cv2.drawMatches(Image1, Image1_key_points, Image2, Image2_key_points, matches[:100], Image2.copy())

scale_percent = 20  # percent of original size
width = int(matched_img.shape[1] * scale_percent / 100)
height = int(matched_img.shape[0] * scale_percent / 100)
dim = (width, height)

# rows = Image2.rows
# cols = Image2.cols

# TEST DE LA ROTATION DE LA PHOTO
img_rot = cv2.warpAffine(Image1, matrix, (
    Image1.shape[1] + 1500, Image1.shape[0] + 1500))  # + 1500 pour avoir l'ensemble de la photo (pas rogné)

resized = cv2.resize(matched_img, dim, interpolation=cv2.INTER_AREA)

scale_percent = 183
width = int(Image1.shape[1] * scale_percent / 100)
height = int(Image1.shape[0] * scale_percent / 100)
dim2 = (width, height)
img_rot = cv2.resize(img_rot, dim2, interpolation=cv2.INTER_AREA)

result = test_overlay(Image2, img_rot, 100, 60)

scale_percent2 = 20  # percent of original size
width2 = int(result.shape[1] * scale_percent2 / 100)
height2 = int(result.shape[0] * scale_percent2 / 100)
dim2 = (width2, height2)
result = cv2.resize(result, dim2, interpolation=cv2.INTER_AREA)

cv2.imshow("Superposition", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
