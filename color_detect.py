import cv2
import numpy as np
from color_trace import*

red_lower = np.array([170, 43, 46])  # 红色下限
red_upper = np.array([180, 255, 255]) # 红色上限

img = cv2.imread('/Users/peterxli/Hualai/detect_object/crop_frame/FFFFFFFFFFFF2185277.jpg')
cv2.imshow('red ball', img)
img_result = color_trace(red_lower, red_upper, img)
cv2.imshow('img_result', img_result)
cv2.waitKey(-1)