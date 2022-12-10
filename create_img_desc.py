import os 
from pathlib import Path
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET


img_list = os.listdir('/Users/jakubrzepkowski/Desktop/Studia/WizjaKomputerowa/Dataset/Kaggle/images')
path_to_img = '/Users/jakubrzepkowski/Desktop/Studia/WizjaKomputerowa/Dataset/Kaggle/images/'
path_to_xml = '/Users/jakubrzepkowski/Desktop/Studia/WizjaKomputerowa/Dataset/Kaggle/annotations/' 
path_to_txt = '/Users/jakubrzepkowski/Desktop/Studia/WizjaKomputerowa/Dataset/Kaggle/correct_annotations/'

for i in img_list:
    img_name = i[:-4]
    # with open(f'{path_to_xml}{img_name}.xml', 'r') as f:
    #     data = f.read()
    f= open(f"{path_to_txt}{img_name}.txt","w+")


    tree = ET.parse(f"{path_to_xml}{img_name}.xml")
    root = tree.getroot()

    width = root.findall(".//size/width")
    height = root.findall(".//size/height")
    xmin = root.findall(".//object/bndbox/xmin")
    xmax = root.findall(".//object/bndbox/xmax")
    ymin = root.findall(".//object/bndbox/ymin")
    ymax = root.findall(".//object/bndbox/ymax")


    for x, xm, y, ym in zip(xmin, xmax, ymin, ymax):
        w_bb = (float(xm.text) - float(x.text))/float(width[0].text)
        h_bb = (float(ym.text) - float(y.text))/float(height[0].text)
        x_center = ((float(xm.text) - float(x.text))/2 + float(x.text))/float(width[0].text)
        y_center = ((float(ym.text) - float(y.text))/2 + float(y.text))/float(height[0].text)
        print(y_center)
        f.write(f"0 {x_center} {y_center} {w_bb} {h_bb}\n")

