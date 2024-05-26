import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

image = cv2.imread(f'../vkr/safety-doors/08_00 МСК 23 мая/point _end/img/img_spvp_c1_t0_raw_0001.png')

points = []

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Left button clicked at ({x}, {y})")
        if len(points) == 2:
            points = []
        points.append((x, y))
    line_thickness = 2
    if len(points) >= 2:
        cv2.line(image, points[0], points[1], (255, 0, 0), thickness=line_thickness)
        cv2.imshow('line', image)

cv2.imshow('line', image)
cv2.setMouseCallback('line', mouse_callback)

key = cv2.waitKey()
while key != 27:
    key = cv2.waitKey()

if len(points) == 2:
    x1, y1 = points[0]
    x2, y2 = points[1]
    print(image.shape)

    x1 /= image.shape[1]
    y1 /= image.shape[0]
    x2 /= image.shape[1]
    y2 /= image.shape[0]
    print(x1, y1)
    print(x2, y2)


    a = (y2-y1)/(x2-x1)
    b = -x2*a + y2
    print(f'y = {a:.4f}x + {b:.4f}')
