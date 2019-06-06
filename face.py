#-*-coding:utf-8 -*-
from collections import OrderedDict
import cv2
import dlib
import numpy as np
from imutils import face_utils
#from matplotlib import pyplot as plt

# Delaunay Triangulation - 턱, 턱 관련 외부 점
def triangle(p, q, m):
    tri = [list(p), list(q), m]
    color = (255, 150, 0)
    # cv2.polylines(output, np.float32([tri]).astype(int), True, color, 2, 16)
    # print(tri)

    return tri


def movetriangle(tri1, x1, y1, x2, y2):
    newt = []
    newt.append([tri1[0][0] + x1, tri1[0][1] + y1])
    newt.append([tri1[1][0] + x2, tri1[1][1] + y2])
    newt.append([tri1[2][0], tri1[2][1]])

    return newt


def move_weight(tri, n):
    if n == 1:
        tri2 = movetriangle(tri, 0, 0, 1, 0)
    elif n == 2:
        tri2 = movetriangle(tri, 1, 0, 2, 0)
    elif n == 3:
        tri2 = movetriangle(tri, 2, 0, 2, 0)
    elif n == 4:
        tri2 = movetriangle(tri, 2, 0, 3, 0)
    elif n == 5:
        tri2 = movetriangle(tri, 3, 0, 4, 0)
    elif n == 6:
        tri2 = movetriangle(tri, 4, 0, 5, 0)
    elif n == 7:
        tri2 = movetriangle(tri, 5, 0, 5, 0)
    elif n == 8:
        tri2 = movetriangle(tri, 5, 0, 0, 0)
    elif n == 9:
        tri2 = movetriangle(tri, 0, 0, -5, 0)
    elif n == 10:
        tri2 = movetriangle(tri, -5, 0, -5, 0)
    elif n == 11:
        tri2 = movetriangle(tri, -5, 0, -4, 0)
    elif n == 12:
        tri2 = movetriangle(tri, -4, 0, -3, 0)
    elif n == 13:
        tri2 = movetriangle(tri, -3, 0, -2, 0)
    elif n == 14:
        tri2 = movetriangle(tri, -2, 0, -2, 0)
    elif n == 15:
        tri2 = movetriangle(tri, -2, 0, -1, 0)
    elif n == 16:
        tri2 = movetriangle(tri, -1, 0, 0, 0)

    return tri2


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
    # print(r1, r)

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r[0]), (t2[i][1] - r[1])))
    # print(t1Rect, t2Rect)
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2Rect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    # img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    # warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    #imgRect = (0.5) * warpImage1 + 0.5 * warpImage1
    tmp =  result[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + warpImage1 * mask
    result[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = tmp


def main(img):

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

    # Ask the detector to find the bounding boxes of each face.
    dets = detector(img_RGB, 0)
    #print("Number of faces detected: {}".format(len(dets)))

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

    righteye = [[]]
    facedir = ''
    for k, d in enumerate(dets):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #    k, d.left(), d.top(), d.right(), d.bottom()))
        facedir = d

        # Get the landmarks/parts for the face in box d.
        shape = predictor(img_RGB, d)

        #for i in range(shape.num_parts):
        #    x = shape.part(i).x
        #    y = shape.part(i).y
        #    cv2.circle(img, (x,y), 3, (255,0,0),-1)

        shape = face_utils.shape_to_np(shape)

        left_eye = [shape[lStart:lEnd]]
        right_eye = [shape[rStart:rEnd]]
        jaw = [shape[jStart:jEnd]]

    #cv2.imshow("point", img)
    #win.add_overlay(dets)

    # Bilateral Filtering
    bImg = cv2.bilateralFilter(img, 8, 60, 60)
    #cv2.imshow("blur", bImg)

    # eye mask
    eye = [right_eye, left_eye]
    eyemask = []
    mask_scaled = []
    center = []
    eye_scale = 1.2
    for i in range(len(eye)):
        # mask 생성
        mask = np.zeros(bImg.shape, dtype=np.uint8)
        roi_corners = np.array(eye[i], dtype=np.int32)

        channel_count = bImg.shape[2] # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,)*channel_count
        cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)
        rEye_mask = cv2.bitwise_and(bImg, mask)
        #cv2.imshow("masked_eye", rEye_mask)

        # mask 1.8배 확대 (눈 주변영역 포함시킴)
        tmp_center = list(map(np.mean,zip(*[eye[i][0][0], eye[i][0][3]])))
        center.append(list(map(int, tmp_center)))
        r, c = bImg.shape[:2]
        M1 = cv2.getRotationMatrix2D(tuple(center[i]), 0, eye_scale+0.2)
        dst1 = cv2.warpAffine(mask, M1, (c, r), borderMode=cv2.BORDER_REFLECT_101)
        #cv2.imshow("mask", dst1)
        eye_mask = cv2.bitwise_and(bImg, dst1)
        #cv2.imshow("masked_scale", eye_mask)

        # masked 눈 확대
        r, c = eye_mask.shape[:2]
        M2 = cv2.getRotationMatrix2D(tuple(center[i]), 0, eye_scale)
        eyemask.append(cv2.warpAffine(eye_mask, M2, (c, r),borderMode=cv2.BORDER_REFLECT_101))
        #cv2.imshow("scale"+str(i), eyemask[i])

        # ROI 확대 (눈 확대한것과 같은 크기로)
        #M3 = cv2.getRotationMatrix2D(tuple(center[i]), 0, eye_scale)
        mask_scaled.append(cv2.warpAffine(dst1, M2, (c, r),borderMode=cv2.BORDER_REFLECT_101))

    eyemask_all = cv2.add(eyemask[0], eyemask[1])
    #cv2.imshow("eyemask_all", eyemask_all)
    mask_all = cv2.add(mask_scaled[0], mask_scaled[1])
    #cv2.imshow("mask_all", mask_all)

    center_eye = list(map(np.mean,zip(*center)))
    center_eye = list(map(int,center_eye))
    #print ('center_eye', center_eye)

    # Clone seamlessly.
    eye_output = cv2.seamlessClone(eyemask_all, bImg, mask_all, tuple(center_eye), cv2.NORMAL_CLONE)

    #cv2.imshow("clone", eye_output)

    # ---------------------------------------------------------------------------------

    jmask = np.zeros(bImg.shape, dtype=np.uint8)
    j_roi_corners = np.array(jaw, dtype=np.int32)

    channel_count = bImg.shape[2] # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(jmask, j_roi_corners, ignore_mask_color)
    jaw_mask = cv2.bitwise_and(bImg, jmask)
    #cv2.imshow("masked", jaw_mask)

    mid = list(map(np.mean, zip(*[jaw[0][1], jaw[0][-2]])))
    mid = list(map(int, mid))

    output = eye_output.copy()

    #imgwarp = np.zeros(output.shape, dtype=output.dtype)
    imgwarp = output.copy()
    #print ('jaw', len(jaw[0]))
    for i in range(1,len(jaw[0])):
        t1 = triangle(jaw[0][i-1], jaw[0][i], mid)
        t2 = move_weight(t1, i)
        color = (0, 254, 30)
        #cv2.polylines(eye_output, np.float32([t1]).astype(int), True, color, 1, 16)

        color = (0, 0, 254)
        #cv2.polylines(eye_output, np.float32([t2]).astype(int), True, color, 1, 16)

        warping(output, imgwarp, t1, t2)

    #cv2.imshow("imgwarp-p", imgwarp)

    face_w = facedir.right() - facedir.left()
    face_h = facedir.bottom() - facedir.top()

    Lx = int(facedir.left()-face_w/5)
    Ly = int(facedir.bottom()+face_h/4)
    Rx = int(facedir.right()+face_w/5)
    Ry = int(facedir.bottom()+face_h/4)

    jaw_op = [Lx, Ly, Rx, Ry]
    for i in range(4):
        if jaw_op[i] < 0:
            jaw_op[i] = 0

    midL = [jaw_op[0], jaw_op[1]]
    midR = [jaw_op[2], jaw_op[3]]

    #outside triangle
    for i in range(1, len(jaw[0])):
        jaw_mid = jaw[0][int(len(jaw[0])/2)][0]
        if jaw[0][i][0] <= jaw_mid:
            t1 = triangle(jaw[0][i - 1], jaw[0][i], midL)
        elif jaw[0][i][0] > jaw_mid:
            t1 = triangle(jaw[0][i - 1], jaw[0][i], midR)
        t2 = move_weight(t1, i)

        color = (250, 30, 30)
        #cv2.polylines(eye_output, np.float32([t1]).astype(int), True, color, 1, 16)
        #print(t1,t2)
        color = (0, 30, 254)
        #cv2.polylines(eye_output, np.float32([t2]).astype(int), True, color, 1, 16)


        warping(output, imgwarp, t1, t2)


    #cv2.imshow("draw", eye_output)
    #cv2.imshow("imgwarp", imgwarp)
    #cv2.imwrite('./output/warp_'+imgname, imgwarp)

    return imgwarp



if __name__ == "__main__":
    select = int(input('Webcam(1) or Image(2): '))

    if select == 1:
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

        while True:
            ret, frame = capture.read()
            cv2.imshow("VideoFrame", frame)
            warpframe = main(frame)
            cv2.imshow("warpframe", warpframe)
            if cv2.waitKey(1) > 0: break

        capture.release()
        cv2.destroyAllWindows()

    elif select == 2:
        imgname = 'w_sq.jpg'
        img = cv2.imread('images/'+imgname)
        warpimg = main(img)
        cv2.imshow("warpimg", warpimg)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

