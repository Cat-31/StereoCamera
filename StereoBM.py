# coding=utf-8
import cv2 as cv
import numpy as np
import time
import os

depthWinTitle = 'Depth'

cv.namedWindow(depthWinTitle)
cv.createTrackbar("num", depthWinTitle, 0, 10, lambda x: None)
cv.createTrackbar("blockSize", depthWinTitle, 5, 255, lambda x: None)
cv.moveWindow("3D Cam", 100, 50)
cv.moveWindow(depthWinTitle, 800, 50)


def callbackfunc(e, x, y, f, p):
    if e == cv.EVENT_LBUTTONDOWN:
        print(threeD[y][x])


cv.setMouseCallback(depthWinTitle, callbackfunc, None)

fs = cv.FileStorage('intrinsics.yml', cv.FILE_STORAGE_READ)
M1 = fs.getNode('M1').mat()
D1 = fs.getNode('D1').mat()
M2 = fs.getNode('M2').mat()
D2 = fs.getNode('D2').mat()

fs = cv.FileStorage('extrinsics.yml', cv.FILE_STORAGE_READ)
R = fs.getNode('R').mat()
T = fs.getNode('T').mat()
R1 = fs.getNode('R1').mat()
P1 = fs.getNode('P1').mat()
R2 = fs.getNode('R2').mat()
P2 = fs.getNode('P2').mat()
Q = fs.getNode('Q').mat()

cap = cv.VideoCapture('http://192.168.43.90:8080/?action=stream')
size = (320, 240)
left_map1, left_map2 = cv.initUndistortRectifyMap(M1, D1, R1, P1, size, cv.CV_16SC2)
right_map1, right_map2 = cv.initUndistortRectifyMap(M2, D2, R2, P2, size, cv.CV_16SC2)

while True:
    ret, frame = cap.read()
    frame = cv.resize(frame, (640, 240), interpolation=cv.CV_8SC1)

    frame_left = frame[0:240, 0:320]
    frame_right = frame[0:240, 320:640]

    left_rectified = cv.remap(frame_left, left_map1, left_map2, cv.INTER_LINEAR)
    right_rectified = cv.remap(frame_right, right_map1, right_map2, cv.INTER_LINEAR)
    rectified = np.concatenate([left_rectified, right_rectified], axis=1)
    image = np.concatenate([frame, rectified])

    imgL = cv.cvtColor(left_rectified, cv.COLOR_BGR2GRAY)
    imgR = cv.cvtColor(right_rectified, cv.COLOR_BGR2GRAY)

    num = cv.getTrackbarPos("num", depthWinTitle)
    blockSize = cv.getTrackbarPos("blockSize", depthWinTitle)
    if blockSize % 2 == 0:
        blockSize += 1
    if blockSize < 5:
        blockSize = 5

    stereo = cv.StereoBM_create(numDisparities=16 * num, blockSize=blockSize)
    print(16 * num, blockSize)
    disparity = stereo.compute(imgL, imgR)

    disp = cv.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    threeD = cv.reprojectImageTo3D(disparity.astype(np.float32) / 16., Q)
    for i in range(1, 30):
        cv.line(image, (0, 16 * i), (640, 16 * i), (0, 255, 0), 1)

    cv.imshow('3D Cam', image)
    cv.imshow(depthWinTitle, disp)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
