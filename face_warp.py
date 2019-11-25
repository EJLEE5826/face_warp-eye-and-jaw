#-*-coding:utf-8 -*-
from collections import OrderedDict
import cv2
import dlib
import numpy as np
from imutils import face_utils
import operator
import keyboard as kb

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

fixed_fill_color = []

# Fix fill color
def define_fixed_fill_color():
    global fixed_fill_color
    fixed_fill_color = []

    for i in range(57):
        fixed_fill_color.append(np.random.normal(size=3))
    fixed_fill_color.append(np.array([-0.5,-0.5,-0.5]))
    # fixed_fill_color.append(np.zeros(3))
    # print(fixed_fill_color)


# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

def coloring_triangle(img, img_orig, tri, alpha, line_color, fill_color=None):
    pt1, pt2, pt3 = tri
    overlay = img.copy()

    tri_r, tri_c = zip(*tri)

    means = int(np.mean(tri_r)), int(np.mean(tri_c))
    # print(tri)
    # print('mean',means, (img_orig[means[1], means[0]]))
    # print(type(img_orig[means[0], means[1]]))
    # color = np.random.choice(255, 3).tolist()
    if fill_color is None:
        fill_color = np.random.normal(size=3)
    center_color = img_orig[means[1], means[0]]
    final_color = center_color + 35.*(fill_color)
    final_color = [c + 50 if c < 10 else c for c in final_color]
    # print('final_color', final_color)

    tri = (np.int32([tri]))
    cv2.fillPoly(overlay, tri, final_color, cv2.LINE_AA)


    cv2.line(overlay, pt1, pt2, line_color, 1, cv2.LINE_AA, 0)
    cv2.line(overlay, pt2, pt3, line_color, 1, cv2.LINE_AA, 0)
    cv2.line(overlay, pt3, pt1, line_color, 1, cv2.LINE_AA, 0)

    transf = alpha
    # cv2.polylines(img, tri, True, delaunay_color, 2)

    cv2.addWeighted(overlay, transf, img, 1 - transf, 0, img)

    # cv2.line(img, pt1, pt2, line_color, 1, cv2.LINE_AA, 0)
    # cv2.line(img, pt2, pt3, line_color, 1, cv2.LINE_AA, 0)
    # cv2.line(img, pt3, pt1, line_color, 1, cv2.LINE_AA, 0)


    return img


# Draw delaunay triangles
def draw_delaunay(img, img_orig, subdiv, line_color, emptypart, alpha=0.5):
    # def is_emptypart():

    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])

    # print(len(triangleList))

    for i, t in enumerate(triangleList):
        isEye = []

        t = list(map(int, t))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        # pt1 = [t[0], t[1]]
        # pt2 = [t[2], t[3]]
        # pt3 = [t[4], t[5]]
        tri = [pt1, pt2, pt3]
        # print(tri)
        # print(eyes)
        for p in tri:
            if p in emptypart:
                isEye.append(True)
            else: isEye.append(False)

        # if nose[0] in tri and nose[1] in tri:
        #     if nose[2] in tri or nose[4] in tri:
        #         pass
        #     else:
        #         continue

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            if not all(isEye):

                # if len(fixed_fill_color) > i:
                #
                # else:
                # fixed_fill_color[i] = coloring_triangle(img, img_orig, tri, alpha, line_color)
                # print(i)
                img = coloring_triangle(img, img_orig, tri, alpha, line_color, fixed_fill_color[i])


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


def face_detection(img):
    # Dlib 인식 위해 BGR->RGB 바꿈
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Face Detection

    # reference: https://bit.ly/2X3Z0Oz

    small_img = cv2.resize(img_gray, (0, 0), fx=0.5, fy=0.5)
    face = faceCascade.detectMultiScale(small_img, scaleFactor=1.05, minNeighbors=5,
                                        minSize=(100,100), flags=cv2.CASCADE_SCALE_IMAGE)

    # define a dictionary that maps the indexes of the facial
    # landmarks to specific face regions

    # faces_folder_path = "./images/female.jpg"

    # 얼굴 인식용 클래스 생성 (기본 제공되는 얼굴 인식 모델 사용)

    # Ask the detector to find the bounding boxes of each face.
    # dets = detector(img_gray, 0)
    # print("Number of faces detected: {}".format(len(dets)))

    # (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    # (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    # (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    # (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

    # if len(dets) == 1:
    # for d in dets:
    #     print("Detection: Left: {} Top: {} Right: {} Bottom: {}".format(
    #        d.left(), d.top(), d.right(), d.bottom()))
    if len(face) == 1:
        for (x, y, w, h) in face:
            x, y, w, h = x*2, y*2, w*2, h*2
            dlib_rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))

            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

            # Get the landmarks/parts for the face in box d.
            shape = predictor(img_gray, dlib_rect)
            # shape = predictor(img_gray, d)

            # for i in range(shape.num_parts):
            #     x = shape.part(i).x
            #     y = shape.part(i).y
            #     cv2.circle(point_img, (x,y), 3, (0,0,255),-1)

            shape = face_utils.shape_to_np(shape).tolist()
            # shape = list(map(tuple, shape))
            # print(shape)
        return shape
    else:
        return None

def del_points(shape):
    # point_img = img.copy()

    # delete too much points
    tmp_shp = []
    del_point = [2, 4, 6, 8, 10, 12, 14, 16, 18,
                 19, 21, 22, 23, 24, 26, 27,
                 29, 30, 33, 35,
                 50, 52, 54, 56, 58, 60, 61, 62, 64, 67, 63, 65, 66, 68]
    for i, p in enumerate(shape):
        if i + 1 not in del_point:
            tmp_shp.append(tuple(p))
            # cv2.circle(point_img, p, 3, (255, 0, 0), -1)
            # cv2.putText(point_img, str(i+1), p, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
            #             color=(255, 255, 255), thickness=1)
            # cv2.putText(point_img, str(len(tmp_shp) - 1), p, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
            #             color=(255, 255, 255), thickness=2)

    # cv2.imshow("point", point_img)
    # cv2.imwrite('./output/point.jpg', point_img)
    return tmp_shp

def move_eye_point(eyes):
    r = [-8, 0, 0, 8, 0, 0]
    c = [0, -10, -10, 0, 5, 5]

    for i, eye in enumerate(eyes):
        eye = list(map(list, eye))
        x, y = zip(*eye)
        x = list(map(operator.add, r, x))
        y = list(map(operator.add, c, y))
        eyes[i] = list(zip(x, y))
    # print(eyes)
    return eyes

def coloring_face(img, shape, ani=False):
    img_orig = img.copy()

    # move eyes points
    l_start, r_start, m_start = 16, 22, 28
    mouth = shape[m_start:m_start + 6]
    nose_bridge = shape[11:16]
    jaw = [shape[1:4], shape[5:8]]

    eyes = [shape[l_start:l_start + 6], shape[r_start:r_start + 6]]
    eyes = move_eye_point(eyes)
    shape[l_start:l_start + 6], shape[r_start:r_start + 6] = eyes

    # Coloring
    size = img.shape
    rect = (0, 0, size[1], size[0])
    line_color = (0, 0, 0)

    subdiv = cv2.Subdiv2D(rect)
    animate = ani

    # fourcc = cv2.VideoWriter_fourcc(*'MP42')
    # video = cv2.VideoWriter('./noise.avi', fourcc, float(24), (1000, 1000))

    # Insert points into subdiv
    for p in shape:
        subdiv.insert(tuple(map(int, p)))

        if animate:
            img_copy = img_orig.copy()

            draw_delaunay(img_copy, img_orig, subdiv, line_color, eyes[0] + eyes[1] + mouth)
            cv2.imshow("animate", img_copy)
            # video.write(img_copy)
            cv2.waitKey(150)

    # video.release()
    # cv2.destroyAllWindows()

    if animate:
        img = img_copy.copy()
        img = coloring_triangle(img, img_orig, nose_bridge[:3], 1, line_color, fixed_fill_color[55])
        img = coloring_triangle(img, img_orig, nose_bridge[:2] + nose_bridge[4:], 1, line_color, fixed_fill_color[56])

        img = coloring_triangle(img, img_orig, jaw[0], 1, line_color, (0,0,0))
        img = coloring_triangle(img, img_orig, jaw[1], 1, line_color, (0,0,0))
        cv2.imshow("animate", img)

    else:
        draw_delaunay(img, img_orig, subdiv, line_color, eyes[0] + eyes[1] + mouth)
        # print('nose', nose_bridge)
        img = coloring_triangle(img, img_orig, nose_bridge[:3], 1, line_color, fixed_fill_color[55])
        img = coloring_triangle(img, img_orig, nose_bridge[:2] + nose_bridge[4:], 1, line_color, fixed_fill_color[56])

        img = coloring_triangle(img, img_orig, jaw[0], 1, line_color,fixed_fill_color[-1])
        img = coloring_triangle(img, img_orig, jaw[1], 1, line_color,fixed_fill_color[-1])

    return img

def main(img, check):
    if check == 1 or kb.is_pressed('s'):
        check = 1
        shape = face_detection(img)

        if shape is None:
            return img, check

        # kb.hook(pressed)
        # kb.add_hotkey(' ', check_press(img, shape))
        shape = del_points(shape)
        img = coloring_face(img, shape)

    # cv2.imshow("delo_img", img)
    # cv2.imwrite('./output/del_img_both.jpg', img)
    # cv2.imwrite("./output/man_eye.jpg",img)

    #win.add_overlay(dets)
    return img, check


    '''
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
    # cv2.imshow("imgwarp", imgwarp)
    #cv2.imwrite('./output/warp_'+imgname, imgwarp)
    return imgwarp
    '''

    # return 0




if __name__ == "__main__":
    select = int(input('Webcam(1) or Video(2) or Image(3): '))
    # select = 3

    define_fixed_fill_color()
    kb.add_hotkey('r',define_fixed_fill_color)

    check = 0

    if select == 1:
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

        while True:
            ret, frame = capture.read()
            #cv2.imshow("VideoFrame", frame)
            warpframe, check = main(frame, check)
            cv2.imshow("warpframe", warpframe)
            if cv2.waitKey(10) == 27: break

        capture.release()
        cv2.destroyAllWindows()

    if select == 2:

        capture = cv2.VideoCapture('./images/input.mp4')

        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('output.avi', fourcc, 18.0, (960, 720))
        while (capture.isOpened()):
            ret, frame = capture.read()

            if ret:
                # cv2.imshow("VideoFrame", frame)
                warpframe, check = main(frame, check)
                cv2.imshow("warpframe", warpframe)
                out.write(warpframe)

                if cv2.waitKey(10) == 27:
                    break

            else:
                break

        capture.release()
        out.release()
        cv2.destroyAllWindows()

    elif select == 3:
        check = 1
        imgname = 'front__man.jpg'
        img = cv2.imread('images/'+imgname)
        warpimg, check= main(img, check)
        cv2.imshow("warpimg", warpimg)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

