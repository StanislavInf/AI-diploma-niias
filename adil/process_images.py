import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from contextlib import redirect_stdout

from ultralytics import YOLOv10

# line_a = 253.1556
# line_b = -120.4453
line_a = -72.4706
line_b = 33.8077
conf = 0.25
inputdir = "../vkr/safety-doors/08_00 МСК 23 мая/point _end/img"
outputdir = "./yolo_out_l_line_l"

# model = YOLOv10(f'./yolov10s.pt', verbose=False)
model = YOLOv10(f'./yolov10l.pt', verbose=False)

def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 5, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    lw / 5,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)

def draw_boxes(image, boxes, labels=[], colors=[], score=True):
    cv2.line(image,
             (int(-line_b/line_a*image.shape[1]), 0),
             (int((1-line_b)/line_a*image.shape[1]), image.shape[0]),
             (0, 0, 255), thickness=2)
    coco_labels = {0: u'__background__', 1: u'person', 2: u'bicycle',3: u'car', 4: u'motorcycle', 5: u'airplane', 6: u'bus', 7: u'train', 8: u'truck', 9: u'boat', 10: u'traffic light', 11: u'fire hydrant', 12: u'stop sign', 13: u'parking meter', 14: u'bench', 15: u'bird', 16: u'cat', 17: u'dog', 18: u'horse', 19: u'sheep', 20: u'cow', 21: u'elephant', 22: u'bear', 23: u'zebra', 24: u'giraffe', 25: u'backpack', 26: u'umbrella', 27: u'handbag', 28: u'tie', 29: u'suitcase', 30: u'frisbee', 31: u'skis', 32: u'snowboard', 33: u'sports ball', 34: u'kite', 35: u'baseball bat', 36: u'baseball glove', 37: u'skateboard', 38: u'surfboard', 39: u'tennis racket', 40: u'bottle', 41: u'wine glass', 42: u'cup', 43: u'fork', 44: u'knife', 45: u'spoon', 46: u'bowl', 47: u'banana', 48: u'apple', 49: u'sandwich', 50: u'orange', 51: u'broccoli', 52: u'carrot', 53: u'hot dog', 54: u'pizza', 55: u'donut', 56: u'cake', 57: u'chair', 58: u'couch', 59: u'potted plant', 60: u'bed', 61: u'dining table', 62: u'toilet', 63: u'tv', 64: u'laptop', 65: u'mouse', 66: u'remote', 67: u'keyboard', 68: u'cell phone', 69: u'microwave', 70: u'oven', 71: u'toaster', 72: u'sink', 73: u'refrigerator', 74: u'book', 75: u'clock', 76: u'vase', 77: u'scissors', 78: u'teddy bear', 79: u'hair drier', 80: u'toothbrush'}
    interesting_labels = {
        0, 1, 2, 3, 4, 5, 9, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31
    }

    for box in boxes:
        if box[-1] in interesting_labels:
            continue

        x1 = box[0]/image.shape[1]
        y1 = box[1]/image.shape[0]
        x2 = box[2]/image.shape[1]
        y2 = box[3]/image.shape[0]
        bbox = [
            (x1, y1),
            (x1, y2),
            (x2, y1),
            (x2, y2)
        ]
        num_points = 0
        for x,y in bbox:
            if (line_a < 0 and y > line_a * x + line_b) or (line_a > 0 and y < line_a * x + line_b):
                num_points += 1
        if num_points == 4:
            color = (255, 0, 0)
            label = "danger"
        else:
            color = (0, 255, 0)
            label = "ok"

        if box[-1] != 0:
            label += f' ({coco_labels[int(box[-1])+1]})'
        label += f' {round(100 * float(box[-2]),1)}%'

        box_label(image, box, label, color)
    return image

def process_image(inputpath, outputpath, conf):
    image = np.asarray(cv2.imread(inputpath))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image, conf=conf)

    image = draw_boxes(image.copy(), results[0].boxes.data)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(outputpath, image)

for filename in sorted(os.listdir(inputdir)):
    inputpath = os.path.join(inputdir, filename)
    outputpath = os.path.join(outputdir, filename)
    # print(inputpath, outputpath)
    process_image(inputpath, outputpath, conf)
