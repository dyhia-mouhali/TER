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


Image1 = cv2.imread("face.jpg")
Image2 = cv2.imread("original.jpg")

Image1_gray = cv2.cvtColor(Image1, cv2.COLOR_BGR2GRAY)  # On grise les images
Image2_gray = cv2.cvtColor(Image2, cv2.COLOR_BGR2GRAY)

sift_initialize = cv2.xfeatures2d.SIFT_create()

Image1_key_points, Image1_descriptors = sift_initialize.detectAndCompute(Image1_gray, None)
Image2_key_points, Image2_descriptors = sift_initialize.detectAndCompute(Image2_gray, None)

print('this is an example of a single SIFT keypoint:\n* * *')
explain_keypoint(Image1_key_points[0])

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(Image1_descriptors, Image2_descriptors, k=2)
#matches = bf.match(Image1_descriptors, Image2_descriptors)
#matches = sorted(matches, key=lambda x: x.distance)

# ratio test, on garde les points d'interet les plus interessants
good = []
for m, n in matches:
    if m.distance < 0.3 * n.distance:
        good.append(m)
        # print("m", m)
        # print("n", n)

print("\nOn garde les points d'interet les plus interessants dans good = ", good)

# Featured matched keypoints from images 1 and 2
pts_img1 = np.float32([Image1_key_points[m.queryIdx].pt for m in good])
pts_img2 = np.float32([Image2_key_points[m.trainIdx].pt for m in good])

print("\n On peut recuperer les points [x y] de seulement la premiere photo :\n", pts_img1)
print("\n On peut recuperer la coordonnee x du premier point enregistre de la premiere photo", pts_img1[0][0])

# On va s'interesser a la deuxieme image c'est sur elle que l'on va rajouter la fenetre englobante
image_fenetre = Image2

# On va chercher le points x min et y min de la fenetre englobante
x_min = pts_img2[0][0]
y_min = pts_img2[0][1]
for x, y in pts_img2:
    if x <= x_min:
        x_min = x
    if y <= y_min:
        y_min = y

# On va chercher le points x max et y max de la fenetre englobante
x_max = pts_img2[0][0]
y_max = pts_img2[0][1]
for x, y in pts_img2:
    if x >= x_max:
        x_max = x
    if y >= y_max:
        y_max = y

cv2.rectangle(image_fenetre, (int(x_min-20), int(y_min-20)), (int(x_max + 150), int(y_max + 150)), (255, 0, 0), 3)

scale_percent = 20  # percent of original size
width = int(image_fenetre.shape[1] * scale_percent / 100)
height = int(image_fenetre.shape[0] * scale_percent / 100)
dim = (width, height)

image_fenetre_resized = cv2.resize(image_fenetre, dim, interpolation=cv2.INTER_AREA)

cv2.imshow("images_fenetre_englobante", image_fenetre_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
