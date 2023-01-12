import torch
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
# from pytesseract import pytesseract
import os
from matplotlib import pyplot as plt
import pandas as pd
import datetime
import pytesseract
import shutil

def create_output_folder(path):
    try:
        os.makedirs(path, exist_ok = True)
        print(f"Directory {path} created successfully")
    except OSError as error:
        print(error)
        print(f"Directory {path} can not be created")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weights_path = Path('./runs/train/yolo_car_plates_test_set/weights/best.pt')
model = torch.hub.load('ultralytics/yolov5', 'custom', weights_path)

df = pd.DataFrame(columns= ['id', 'date', 'path', 'car_photo_path', 'plate_img_path', 'licence_plate_number', 'anonymous_plate', 'anonymous_number'])

output_path = os.path.join('.', 'results')
try:
    shutil.rmtree(output_path)
    print(f"Directory {output_path} has been deleted")
except:
    print(f"Directory {output_path} dose not exist")

output_path_photos = os.path.join('.', 'results', 'cars')
create_output_folder(output_path_photos)

output_path_plates = os.path.join('.', 'results', 'plates')
create_output_folder(output_path_plates)

output_path_plates_text = os.path.join('.', 'results', 'plates_text')
create_output_folder(output_path_plates_text)

output_path_plates_anon = os.path.join('.', 'results', 'plates_anon')
create_output_folder(output_path_plates_anon)

output_path_plates_text_anon = os.path.join('.', 'results', 'plates_text_anon')
create_output_folder(output_path_plates_text_anon)

dataset_path = os.path.join('.', 'Dataset', 'Kaggle', 'images', 'test')
# dataset_path = os.path.join('.', 'inferention_dataset')

id = 0
for file in os.listdir(dataset_path):
    if file.endswith(".jpg"):
        img_path = os.path.join(dataset_path, file)
        
        img = cv2.imread(img_path)

        #przepuszczanie przez YOLOv5 -> wyznaczenie położenia tablicy
        results = model(img)

        boxes = results.pandas().xyxy[0].sort_values('xmin')

        #wyznaczenie gdzie jest najwięszka pewność, że tablica się znajduje (bo często są dwa albo 3 napisy na zdjęciach)
        confidence = 0
        for i in range(len(boxes)):
            #bierzemy tylko tablicę z największą pewnością 
            if boxes['confidence'][i] > confidence:
                confidence = boxes['confidence'][i]
                y_min = int(boxes['ymin'][i])
                x_min = int(boxes['xmin'][i])
                y_max = int(boxes['ymax'][i])
                x_max = int(boxes['xmax'][i])
        
        car_img = img.copy()
        cv2.putText(car_img, f"Conf = {round(confidence, 3)}", (x_min + 10, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(car_img, (x_min, y_min), (x_max, y_max), color = (0, 0, 255), thickness = 2)
        car_image_path = os.path.join(output_path_photos,  (str(id) + '_car_photo_' + file))
        cv2.imwrite(car_image_path, car_img)

        #wycinanie tylko tablicy
        croped_plate = img[y_min:y_max, x_min:x_max]

        #przeskalowanie
        scale_percent = 500 # o ile procent skalujemy - 500 %
        width = int(croped_plate.shape[1] * scale_percent / 100)
        height = int(croped_plate.shape[0] * scale_percent / 100)
        dim = (width, height)
        croped_plate_resized = cv2.resize(croped_plate, dim, interpolation = cv2.INTER_AREA)
        plate_img_path = os.path.join(output_path_plates,  (str(id) + '_plate_photo_' + file))
        cv2.imwrite(plate_img_path, croped_plate_resized)

        plate_to_recognize = croped_plate_resized.copy()
        cv2norm_img = np.zeros((plate_to_recognize.shape[0], plate_to_recognize.shape[1]))
        plate_to_recognize = cv2.normalize(plate_to_recognize, plate_to_recognize, 0, 255, cv2.NORM_MINMAX)
        plate_to_recognize = cv2.threshold(plate_to_recognize, 100, 255, cv2.THRESH_BINARY)[1]
        plate_to_recognize = cv2.GaussianBlur(plate_to_recognize, (1, 1), 0)
        #TODO dodać config jakie znaki można rozpoznawać
        plate_text = pytesseract.image_to_string(plate_to_recognize)
        
        temp_file = file.replace('.jpg', '.txt')
        plate_text_path = os.path.join(output_path_plates_text,  (str(id) + '_plate_text_' + temp_file))
        plate_text_path = plate_text_path.replace("\n", '')
        plate_text_path = plate_text_path.replace(" ", '')
        with open(plate_text_path, 'w') as f:
            f.write(f"recognizet plate text: {plate_text}")

        #TODO anon plates and save it into plates_text_anon
        # output_path_plates_text_anon

        img_gray_lp = cv2.cvtColor(croped_plate_resized, cv2.COLOR_BGR2GRAY)                        #to gray scale
        _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  #convert to binary image
        img_binary_lp = cv2.erode(img_binary_lp, (3,3))
        img_binary_lp = cv2.dilate(img_binary_lp, (3,3))
        
        # TODO
        #make borders white - jak starczy czasu

        LP_WIDTH = img_binary_lp.shape[0]
        LP_HEIGHT = img_binary_lp.shape[1]
 
        #estymacja wielkości konturu znaków 
        lower_width = LP_WIDTH/6
        upper_width = LP_WIDTH/2
        lower_height = LP_HEIGHT/10
        upper_height = 2*LP_HEIGHT/3

        temp_img = img_binary_lp.copy()
        blured_plate = img_binary_lp.copy()

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

                # extracting each character using the enclosing rectangle's coordinates.
                char = temp_img[intY:intY+intHeight, intX:intX+intWidth]
                char = cv2.resize(char, (20, 40))

                cv2.rectangle(blured_plate, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)

                # ukrywanie 4 znaków na tablicy 
                if i > 4:
                    cv2.rectangle(blured_plate, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), -1)
                
                char = cv2.subtract(255, char)
                char_copy = np.zeros((44,24))
                char_copy[2:42, 2:22] = char
                char_copy[0:2, :] = 0
                char_copy[:, 0:2] = 0
                char_copy[42:44, :] = 0
                char_copy[:, 22:24] = 0
                
                #TODO dodać połączenie między daną literą a lokalizacją na tablicy  
                img_res.append(char_copy) # List that stores the character's binary image (unsorted)

        blured_plate_path = os.path.join(output_path_plates_anon,  (str(id) + '_blured_plate_' + file))
        cv2.imwrite(blured_plate_path, blured_plate)

        #zczytanie akturalnej daty
        ct = datetime.datetime.now()

        #zapisanie danych do bazy danych
        # (columns= ['id', 'date', 'path', 'car_photo_path', 'plate_img_path', 'licence_plate_number', 'anonymous_plate', 'anonymous_number'])
        # parking_database_row = [id, ct, img_path, car_image_path, plate_img_path, plate_text_path, blured_plate_path, ]
        id += 1 

