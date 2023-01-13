import os
from pathlib import Path
import shutil

images = os.listdir('Dataset/Kaggle/images')
img_path = Path('Dataset/Kaggle/images')
label_path = Path('Dataset/Kaggle/labels')

png = os.listdir('Dataset/Kaggle/images/val/')


file_names = []
for name in png:
    os.rename(f'Dataset/Kaggle/images/val/{name}', f'Dataset/Kaggle/images/val/{name[:-4]}.jpg')

x_train, x_val, y_train, y_val = [], [], [], []


for i,data in enumerate(file_names):
    if i%4 == 0:
        x_val.append(f'{data}.png')
        y_val.append(f'{data}.txt')
    else:
        x_train.append(f'{data}.png')
        y_train.append(f'{data}.txt')
    print(i)



for x, y in zip(x_val, y_val):
    shutil.move(img_path/x, img_path/'val'/x)
    shutil.move(label_path/y, label_path/'val'/y)

for x, y in zip(x_train, y_train):
    shutil.move(img_path/x, img_path/'train'/x)
    shutil.move(label_path/y, label_path/'train'/y)
