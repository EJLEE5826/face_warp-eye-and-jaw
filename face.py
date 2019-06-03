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
    jaw = [shape[jStart+1:jEnd-1]]

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
    cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)
    #rEye_mask = cv2.bitwise_and(bImg, mask)
    #cv2.imshow("masked", masked_image0)

    # mask 1.8배 확대 (눈 주변영역 포함시킴)
    tmp_center = list(map(np.mean,zip(*[eye[i][0][0], eye[i][0][3]])))
    center.append(list(map(int, tmp_center)))
    r, c = bImg.shape[:2]
    M1 = cv2.getRotationMatrix2D(tuple(center[i]), 0, 1.3)
    dst1 = cv2.warpAffine(mask, M1, (c, r), borderMode=cv2.BORDER_REFLECT_101)
    #cv2.imshow("mask", dst1)
    eye_mask = cv2.bitwise_and(bImg, dst1)
    #cv2.imshow("masked_scale", rEye_mask)

    # masked 눈 확대
    r, c = eye_mask.shape[:2]
    M2 = cv2.getRotationMatrix2D(tuple(center[i]), 0, 1.3)
    eyemask.append(cv2.warpAffine(eye_mask, M2, (c, r),borderMode=cv2.BORDER_REFLECT_101))
    #cv2.imshow("scale"+str(i), eyemask[i])

    # ROI 확대 (눈 확대한것과 같은 크기로)
    M3 = cv2.getRotationMatrix2D(tuple(center[i]), 0, 1.3)
    mask_scaled.append(cv2.warpAffine(dst1, M3, (c, r),borderMode=cv2.BORDER_REFLECT_101))

eyemask_all = cv2.add(eyemask[0], eyemask[1])
#cv2.imshow("eyemask_all", eyemask_all)
mask_all = cv2.add(mask_scaled[0], mask_scaled[1])
#cv2.imshow("mask_all", mask_all)

print(center[0], center[1])
center_eye = list(map(np.mean,zip(*center)))
center_eye = list(map(int,center_eye))
#print ('center_eye', center_eye)

# Clone seamlessly.
eye_output = cv2.seamlessClone(eyemask_all, bImg, mask_all, tuple(center_eye), cv2.NORMAL_CLONE)

cv2.imshow("clone", eye_output)



# ---------------------------------------------------------------------------------

jmask = np.zeros(bImg.shape, dtype=np.uint8)
j_roi_corners = np.array(jaw, dtype=np.int32)

channel_count = bImg.shape[2] # i.e. 3 or 4 depending on your image
ignore_mask_color = (255,)*channel_count
cv2.fillPoly(jmask, j_roi_corners, ignore_mask_color)
jaw_mask = cv2.bitwise_and(bImg, jmask)
cv2.imshow("masked", jaw_mask)

mid = list(map(np.mean, zip(*[jaw[0][0], jaw[0][14]])))
mid = list(map(int, mid))
#print('mid', mid)

output = eye_output.copy()

# Delaunay Triangulation - 턱, 턱 관련 외부 점
def triangle(p, q, m):
    tri = [list(p), list(q), m]
    color = (255, 150, 0)
    #cv2.polylines(output, np.float32([tri]).astype(int), True, color, 2, 16)
    print(tri)

    return tri

# 되는지 보려고 임의로 그냥 옮긴점
def movetriangle(tri1):
    newt = []
    for i in range(3):
        newt.append([tri1[i][0], tri1[i][1]-10])

    color = (0, 150, 0)
    #cv2.polylines(output, np.float32([newt]).astype(int), True, color, 2, 16)

    return newt


def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


def warping(img1, result, t1, t2):

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r[0]), (t2[i][1] - r[1])))

    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t1Rect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    #img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    #warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    imgRect = (0.5) * warpImage1 + 0.5 * warpImage1

    result[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = result[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * ( 1 - mask ) + imgRect * mask


#imgwarp = np.zeros(output.shape, dtype=output.dtype)
imgwarp = output.copy()

for i in range(1,len(jaw[0])):
    t1 = triangle(jaw[0][i-1], jaw[0][i], mid)
    t2 = movetriangle(t1)

    warping(output, imgwarp, t1, t2)


#cv2.imshow("draw", output)
cv2.imshow("imgwarp", imgwarp)
# 점 줄어드는거 어떻게구하지.. 밑에꺼 참고해서?
'''
def euclideanDist(p, q):
    Point diff = p - q;
    return cv2.sqrt(diff.x*diff.x + diff.y*diff.y);

def GetControlPointWeight():
    std::vector<double> IDW::GetControlPointWeight(cv::Point input)
{
    std::vector<double> weightMap;
    double weightSum = 0;
    for (int i = 0; i < startControlPoint_.size(); i++)
    {
        double temp = 1 / (Distance(endControlPoint_[i], input) + EPS);
        temp = pow(temp, weight_);
        weightSum = weightSum + temp;
        weightMap.push_back(temp);
    }
    for (int i = 0; i < startControlPoint_.size(); i++)
    {
        weightMap[i] /= weightSum;
    }
    return weightMap;
}
'''

# bounding box 구하고 해당 영역에 대해서 affinetransform

# Copy triangular region of the rectangular patch to the output image


# dlib.hit_enter_to_continue()
cv2.waitKey(0)
cv2.destroyAllWindows()
