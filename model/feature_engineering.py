import os
import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt

default_directory = r"F:\Celebrity Face Recognition\model"
os.chdir(default_directory)

# thanks to StackOverflow
def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255;
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H

path_to_cr_data = '../dataset/cropped/'
cropped_image_dirs = []
celebrity_file_names_dict = {}

for cr_img_dir in os.scandir(path_to_cr_data):
    cropped_image_dirs.append(cr_img_dir)
    celebrity_name = cr_img_dir.path.split('/')[-1]
    celebrity_file_names_dict[celebrity_name] = []
    for cr_img_file_path in os.scandir(cr_img_dir):
        celebrity_file_names_dict[celebrity_name].append(cr_img_file_path)
# print(celebrity_file_names_dict)

class_dict = {}
count = 0
for celebrity_name in celebrity_file_names_dict.keys():
    class_dict[celebrity_name] = count
    count = count + 1
# print(class_dict)

X, y = [], []
for celebrity_name, training_files in celebrity_file_names_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img,'db1',5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
        X.append(combined_img)
        y.append(class_dict[celebrity_name])
# print(len(X))
# print(y)

X = np.array(X).reshape(len(X),4096).astype(float)
print(X.shape)

