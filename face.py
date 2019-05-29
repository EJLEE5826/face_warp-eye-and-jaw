#-*-coding:utf-8 -*-
from collections import OrderedDict
import cv2
import dlib
import numpy as np
from imutils import face_utils
from matplotlib import pyplot as plt

img = cv2.imread('images/female.jpg')

# Dlib 인식 위해 BGR->RGB 바꿈
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Face Detection

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
predictor_path = "shape_predictor_68_face_landmarks.dat"
faces_folder_path = "./images/female.jpg"

# 얼굴 인식용 클래스 생성 (기본 제공되는 얼굴 인식 모델 사용)
detector = dlib.get_frontal_face_detector()
# 인식된 얼굴에서 랜드마크 찾기위한 클래스 생성
predictor = dlib.shape_predictor(predictor_path)
#win = dlib.image_window()
#cv2.namedWindow('Face')

#fimg = #dlib.load_rgb_image(faces_folder_path)
#win.clear_overlay()
#win.set_image(img_RGB)

# Ask the detector to find the bounding boxes of each face.
dets = detector(img_RGB, 1)
print("Number of faces detected: {}".format(len(dets)))

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

righteye = [[]]
for k, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))

    # Get the landmarks/parts for the face in box d.
    shape = predictor(img_RGB, d)

    # Draw the face landmarks on the screen.
    #win.add_overlay(shape)

    shape = face_utils.shape_to_np(shape)

    left_eye = [shape[lStart:lEnd]]
    right_eye = [shape[rStart:rEnd]]
    jaw = [shape[jStart:jEnd]]

#win.add_overlay(dets)

# Bilateral Filtering
bImg = cv2.bilateralFilter(img, 9, 75, 75)
cv2.imshow("blur", bImg)

'''
mask = np.zeros(blurdst.shape, dtype=np.uint8)
roi_corners = np.array(right_eye+left_eye, dtype=np.int32)
'''

# eye mask
eye = [right_eye, left_eye]
eyemask = []
mask_scaled = []
center = []
for i in range(len(eye)):
    # mask 생성
    mask = np.zeros(bImg.shape, dtype=np.uint8)
    roi_corners = np.array(eye[i], dtype=np.int32)

    channel_count = bImg.shape[2] # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    #rEye_mask = cv2.bitwise_and(bImg, mask)
    #cv2.imshow("masked", masked_image0)

    # mask 1.8배 확대 (눈 주변영역 포함시킴)
    tmp_center = list(map(np.mean,zip(*[eye[i][0][0], eye[i][0][3]])))
    center.append(list(map(int, tmp_center)))
    r, c = bImg.shape[:2]
    M1 = cv2.getRotationMatrix2D(tuple(center[i]), 0, 1.3)
    dst1 = cv2.warpAffine(mask, M1, (c, r))
    #cv2.imshow("mask", dst1)
    eye_mask = cv2.bitwise_and(bImg, dst1)
    #cv2.imshow("masked_scale", rEye_mask)

    # masked 눈 확대
    r, c = eye_mask.shape[:2]
    M2 = cv2.getRotationMatrix2D(tuple(center[i]), 0, 1.3)
    eyemask.append(cv2.warpAffine(eye_mask, M2, (c, r)))
    #cv2.imshow("scale"+str(i), eyemask[i])

    # ROI 확대
    M3 = cv2.getRotationMatrix2D(tuple(center[i]), 0, 1.3)
    mask_scaled.append(cv2.warpAffine(dst1, M3, (c, r)))

eyemask_all = cv2.add(eyemask[0], eyemask[1])
cv2.imshow("eyemask_all", eyemask_all)
mask_all = cv2.add(mask_scaled[0], mask_scaled[1])
cv2.imshow("mask_all", mask_all)

print(center[0], center[1])
center_All = list(map(np.mean,zip(*center)))
center_All = list(map(int,center_All))
print (center_All)

# Clone seamlessly.
output = cv2.seamlessClone(eyemask_all, bImg, mask_all, tuple(center_All), cv2.NORMAL_CLONE)

cv2.imshow("clone", output)

# dlib.hit_enter_to_continue()
cv2.waitKey(0)
cv2.destroyAllWindows()
