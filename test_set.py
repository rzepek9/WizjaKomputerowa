import os
import shutil
name = os.listdir('/home/s175668/wizjakomputerowa/WizjaKomputerowa/Dataset/Kaggle/images/test')
path = '/home/s175668/wizjakomputerowa/WizjaKomputerowa/Dataset/Kaggle/images/val'
test = '/home/s175668/wizjakomputerowa/WizjaKomputerowa/Dataset/Kaggle/images/test'
label = '/home/s175668/wizjakomputerowa/WizjaKomputerowa/Dataset/Kaggle/labels/val'
label_test = '/home/s175668/wizjakomputerowa/WizjaKomputerowa/Dataset/Kaggle/labels/test'
for i in name:
    
    if '.txt' in i:
        shutil.move(f'{test}/{i}', f'{label_test}/{i}')