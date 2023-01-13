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
import time

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

# dataset_path = os.path.join('.', 'Dataset', 'Kaggle', 'images', 'test')
dataset_path = os.path.join('.', 'inferention_dataset')

delays = []
id = 0
for file in os.listdir(dataset_path):
    start = time.time()
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
        plate_to_recognize = cv2.cvtColor(plate_to_recognize, cv2.COLOR_BGR2GRAY) 
        plate_to_recognize = cv2.normalize(plate_to_recognize, plate_to_recognize, 0, 255, cv2.NORM_MINMAX)
        plate_to_recognize = cv2.threshold(plate_to_recognize, 100, 255, cv2.THRESH_BINARY)[1]
        plate_to_recognize = cv2.GaussianBlur(plate_to_recognize, (1, 1), 0)
        
        plate_text = pytesseract.image_to_string(plate_to_recognize, config='--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPRSTUWXYZ0123456789')
        
        temp_file = file.replace('.jpg', '.txt')
        plate_text_path = os.path.join(output_path_plates_text,  (str(id) + '_plate_text_' + temp_file))
        plate_text = plate_text.replace("\n", '')
        plate_text = plate_text.replace(" ", '')
        with open(plate_text_path, 'w') as f:
            f.write(f"recognizet plate text: {plate_text}")


        anon_plate_text_temp = list(plate_text)
        try:
            anon_plate_text_temp[0] = chr(36)
        except:
            pass
        try:
            anon_plate_text_temp[2] = chr(36)
        except:
            pass
        try:
            anon_plate_text_temp[4] = chr(36)
        except:
            pass
        anon_plate_text = ''.join(anon_plate_text_temp)
        output_path_plates_text_anon_full = os.path.join(output_path_plates_text_anon,  (str(id) + '_plate_text_' + temp_file))
        with open(output_path_plates_text_anon_full, 'w') as f:
            f.write(f"recognizet plate text: {anon_plate_text}")

        img_gray_lp = cv2.cvtColor(croped_plate_resized, cv2.COLOR_BGR2GRAY)                        #to gray scale
        _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  #convert to binary image
        img_binary_lp = cv2.erode(img_binary_lp, (3,3))
        img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

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
        cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)

        x_list = []
        y_list = []
        width_list = []
        height_list = []

        for i, cntr in enumerate(cntrs) :
            intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

            # sprawdzenie czy rozmair estymowanych prostokontów ze znakami jest logiczny
            if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
                x_list.append(intX)
                y_list.append(intY)
                width_list.append(intWidth)
                height_list.append(intHeight)

                cv2.rectangle(blured_plate, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
        
        x_list_sorted = sorted(x_list)
        y_list_sorted = [x for _, x in sorted(zip(x_list, y_list), key= lambda pair: pair[0])]
        width_list_sorted = [x for _, x in sorted(zip(x_list, width_list), key= lambda pair: pair[0])]
        height_list_sorted = [x for _, x in sorted(zip(x_list, height_list), key= lambda pair: pair[0])]

        try:
            intX, intY, intWidth, intHeight = x_list_sorted[0], y_list_sorted[0], width_list_sorted[0], height_list_sorted[0]
            cv2.rectangle(blured_plate, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), -1)
        except:
            pass
        try:
            intX, intY, intWidth, intHeight = x_list_sorted[2], y_list_sorted[2], width_list_sorted[2], height_list_sorted[2]
            cv2.rectangle(blured_plate, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), -1)
        except:
            pass
        try:
            intX, intY, intWidth, intHeight = x_list_sorted[4], y_list_sorted[4], width_list_sorted[4], height_list_sorted[4]
            cv2.rectangle(blured_plate, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), -1)
        except:
            pass

        blured_plate_path = os.path.join(output_path_plates_anon,  (str(id) + '_blured_plate_' + file))
        cv2.imwrite(blured_plate_path, blured_plate)



        #zczytanie akturalnej daty
        ct = datetime.datetime.now()

        #zapisanie danych do bazy danych
        df.loc[len(df.index)] = [id, ct, img_path, car_image_path, plate_img_path, plate_text, blured_plate_path, anon_plate_text]
        id += 1 

        end = time.time()
        delays.append(end - start)

print(df)
print('\n')
print(f"Średni czas inferencji: {np.mean(delays)}")
