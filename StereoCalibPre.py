# coding=utf-8
import cv2
import numpy as np
import os


def find_chessboard_corners(img):
    h, w = img.shape[:2]
    assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
    pattern_size = (8, 6)
    found, corners = cv2.findChessboardCorners(img, pattern_size)
    vis = None
    if found:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(vis, pattern_size, corners, found)

    return found, vis


cap = cv2.VideoCapture('http://192.168.43.90:8080/?action=stream')

firstFrame = None
window_size = 0
i = 0
lst = open('./data/imglst.lst', 'w')

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 240), interpolation=cv2.CV_8SC1)
    cv2.imshow('img', frame)

    frame_left = frame[0:240, 0:320]
    frame_right = frame[0:240, 320:640]
    found1, vis1 = find_chessboard_corners(cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY))
    found2, vis2 = find_chessboard_corners(cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY))

    if found1 & found2:
        image = np.concatenate([vis1, vis2], axis=1)
        image = np.concatenate([frame, image])
        cv2.imwrite(('./data/%dleft.jpg' % i), frame_left)
        cv2.imwrite(('./data/%dright.jpg' % i), frame_right)
        cv2.imshow('output', image)
        lst.writelines(["\"" + os.getcwd() + ('/data/%dleft.jpg' % i) + '\"\n'])
        lst.writelines(["\"" + os.getcwd() + ('/data/%dright.jpg' % i) + '\"\n'])
        i = i + 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

lst.close()
cap.release()
cv2.destroyAllWindows()
