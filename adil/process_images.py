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
outputdir = "./yolo_out_x_2_points"

# model = YOLOv10(f'./yolov10s.pt', verbose=False)
# model = YOLOv10(f'./yolov10l.pt', verbose=False)
model = YOLOv10(f'./yolov10x.pt', verbose=False)

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

    for box in boxes:
        if box[-1] != 0:
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
        centroid = ((x1+x2)/2, (y1+y2)/2)

        # condition = (num_points == 4)
        # condition = (line_a < 0 and centroid[1] > line_a * centroid[0] + line_b) or (line_a > 0 and centroid[1] < line_a * centroid[0] + line_b)
        condition = (num_points >= 2)
        if condition:
            color = (255, 0, 0)
            label = "danger"
        else:
            color = (0, 255, 0)
            label = "ok"

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
