import torch
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from pytesseract import pytesseract
import os
from matplotlib import pyplot as plt

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


img_path = './Dataset/Kaggle/images/test/Cars96.jpg'

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

# cv2.imshow('image', img)
# cv2.waitKey()

#wycinanie tylko tablicy
croped_plate = img[y_min:y_max, x_min:x_max]


#przeskalowanie
scale_percent = 500 # percent of original size
width = int(croped_plate.shape[1] * scale_percent / 100)
height = int(croped_plate.shape[0] * scale_percent / 100)
dim = (width, height)
croped_plate_resized = cv2.resize(croped_plate, dim, interpolation = cv2.INTER_AREA)

# filename = 'croped_plate.png'
# cv2.imwrite(filename, croped_plate)

img_gray_lp = cv2.cvtColor(croped_plate_resized, cv2.COLOR_BGR2GRAY)                        #to gray scale
_, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  #convert to binary image
img_binary_lp = cv2.erode(img_binary_lp, (3,3))
img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

LP_WIDTH = img_binary_lp.shape[0]
LP_HEIGHT = img_binary_lp.shape[1]

# TODO
#make borders white

# Estimations of character contours sizes of cropped license plates
lower_width = LP_WIDTH/6
upper_width = LP_WIDTH/2
lower_height = LP_HEIGHT/10
upper_height = 2*LP_HEIGHT/3

# cv2.imshow('croped', img_binary_lp)
# cv2.waitKey()
# cv2.imwrite('contour.jpg',img_binary_lp)
temp_img = img_binary_lp.copy()
ii = img_binary_lp.copy()

cntrs, _ = cv2.findContours(temp_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

x_cntr_list = []
target_contours = []
img_res = []

for i, cntr in enumerate(cntrs) :
    intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

    # checking the dimensions of the contour to filter out the characters by contour's size
    if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
        x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

        char_copy = np.zeros((44,24))
        # extracting each character using the enclosing rectangle's coordinates.
        char = temp_img[intY:intY+intHeight, intX:intX+intWidth]
        char = cv2.resize(char, (20, 40))

        # temp usage
        # path_to_tesseract = "C:/Program Files/Tesseract-OCR/tesseract.exe"
        # pytesseract.tesseract_cmd = path_to_tesseract
        # text = pytesseract.image_to_string(char, lang ='eng',
        #     config ='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        # print(f'rozpoznano {text}')

        cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
        if i % 2 == 0:
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), -1)
        
        img_res.append(char_copy) # List that stores the character's binary image (unsorted)

cv2.imshow('Predicted segments', ii)
cv2.waitKey()

indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
img_res_copy = []
for idx in indices:
    img_res_copy.append(img_res[idx])# stores character images according to their index
img_res = np.array(img_res_copy)



