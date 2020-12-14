import numpy as np
import cv2
import dlib
import time
import os
import glob

detector = dlib.simple_object_detector('detector.svm')
# cap = cv2.VideoCapture('test5.mp4')
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap = cv2.VideoCapture(0)
# current_path = os.getcwd()
# test_folder = current_path + '/20200819/'
test_folder = '/home/hualai/ponder/lxy_project/light_detect/dlib-19.21/tools/imglab/build/0820photo'
# def testvideo():
#     while cap.isOpened():
#         ret,frame = cap.read()
#         frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
#         wheel_rects = detector(frame,0)
#         start = time.time()
#         for index,wheel in enumerate(wheel_rects):
#             print('wheel{};left{};top{};right{};bottom{}'.format(index,wheel.left(),wheel.top(),wheel.right(),wheel.bottom()))
#             left = wheel.left()
#             top = wheel.top()
#             right = wheel.right()
#             bottom = wheel.bottom()
#             cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),3)
#         cv2.imshow('demo',frame)
#         end = time.time()
#         print (end-start)
#         if cv2.waitKey(100) & 0xFF == ord('q'):
#             break

def testphoto():
    for f in glob.glob(test_folder+'*.jpg'):
        start = time.time()
        print("Processing file: {}".format(f))
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        b, g, r = cv2.split(img)
        img2 = cv2.merge([r, g, b])
        dets = detector(img2)
        print("Number of faces detected: {}".format(len(dets)))
        for index, board in enumerate(dets):
            print('board {}; left {}; top {}; right {}; bottom {}'.format(index, board.left(), board.top(), board.right(), board.bottom()))
            left = board.left()
            top = board.top()
            right = board.right()
            bottom = board.bottom()
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
            cv2.namedWindow(f, cv2.WINDOW_AUTOSIZE)
            end = time.time()
            print(end - start)
            cv2.imshow(f, img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def main():
    testphoto()
    # testvideo()

if __name__ == '__main__':
    main()


