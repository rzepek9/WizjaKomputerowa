import torch
from pathlib import Path
import numpy as np
import cv2
import pytesseract


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


weights_path = Path('./runs/train/yolo_car_plates_test_set/weights/best.pt')

# YOLOv5 root directory
data = './data/kaggle.yaml'

dnn=False
half=False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.hub.load('ultralytics/yolov5', 'custom', './runs/train/yolo_car_plates_test_set/weights/best.pt')

# dataloader.py
img_path = './Dataset/Kaggle/images/test/Cars27.jpg'

img = cv2.imread(img_path)
results = model(img)

boxes = results.pandas().xyxy[0].sort_values('xmin')
# print(boxes)

confidence = 0
for i in range(len(boxes)):
    #bierzemy tylko tablicę z największą pewnością 
    if boxes['confidence'][i] > confidence:
        confidence = boxes['confidence'][i]
        y_min = int(boxes['ymin'][i])
        x_min = int(boxes['xmin'][i])

        y_max = int(boxes['ymax'][i])
        x_max = int(boxes['xmax'][i])

        
color = (0, 0, 255)
thickness = 2

cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)

cv2.imshow('image', img)
cv2.waitKey()

#wycinanie tylko tablicy
croped_plate = img[y_min:y_max, x_min:x_max]

#przeskalowanie
scale_percent = 300 # percent of original size
width = int(croped_plate.shape[1] * scale_percent / 100)
height = int(croped_plate.shape[0] * scale_percent / 100)
dim = (width, height)

# croped_plate = cv2.resize(croped_plate, dim, interpolation = cv2.INTER_AREA)
cv2.imshow('croped', croped_plate)
cv2.waitKey()

# to trzeba sobie pobrać
# https://github.com/UB-Mannheim/tesseract/wiki
#tu trzeba zmienić na swoje
# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
# pytesseract.pytesseract.tesseract_cmd  = "C:/Program Files/Tesseract-OCR/tesseract.exe"

plate = pytesseract.image_to_string(croped_plate)
print(f"Teks na tablicy: {plate}")