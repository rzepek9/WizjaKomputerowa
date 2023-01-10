from models.common import DetectMultiBackend

import torch
from models.common import DetectMultiBackend
from pathlib import Path
import numpy as np
import cv2
from PIL import Image


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


weights_path = Path('/home/s175668/wizjakomputerowa/WizjaKomputerowa/runs/train/yolo_car_plates_test_set/weights/best.pt')

  # YOLOv5 root directory
data = '/home/s175668/wizjakomputerowa/WizjaKomputerowa/data/kaggle.yaml'

dnn=False
half=False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.hub.load('ultralytics/yolov5', 'custom', '/home/s175668/wizjakomputerowa/WizjaKomputerowa/runs/train/yolo_car_plates_test_set/weights/best.pt')
# model = DetectMultiBackend(weights_path, device=device, dnn=dnn, data=data, fp16=half)
# stride, names, pt = model.stride, model.names, model.pt

# dataloader.py
img = cv2.imread('/home/s175668/wizjakomputerowa/WizjaKomputerowa/Dataset/Kaggle/images/test/Cars19.jpg')
im = Image.open('/home/s175668/wizjakomputerowa/WizjaKomputerowa/Dataset/Kaggle/images/test/Cars19.jpg')
# im = letterbox(img, 640, stride=stride, auto=pt)[0]  # padded resize
# im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
# im = np.ascontiguousarray(im)
results = model(img)

# detect.py
# im = torch.from_numpy(im).to(model.device)
# im = im.float()
# im /= 255
# if len(im.shape) == 3:
#     im = torch.unsqueeze(im, 0)
# im = im.float()

# output = model(im)
im1 = im.crop((147.17978, 194.22554, 222.13144, 231.72839))
im = im.convert('RGB')
im.save("tablica.jpg")
print(results.xyxy[0])
print("model załadowany")