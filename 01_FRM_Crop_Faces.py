import numpy as np
import cv2
from glob import glob
import matplotlib
matplotlib.use('TkAgg')  # Switch to an interactive backend
import matplotlib.pyplot as plt



fpath = glob("./data/female/*.jpg")
mpath = glob("./data/male/*.jpg")
haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')


print(f"Number of images in the female folder is {len(fpath)}")
print(f"Number of images in the male folder is {len(mpath)}")

for i in range(len(mpath)):
    try:
        # Step 1: Read the image and convert to RGB
        img = cv2.imread(mpath[i])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert the bgr Image to RGB
        # Step 2: Apply Haar Cascade Classifier
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        faces_list = haar.detectMultiScale(gray, 1.5, 5)
        for x,y,w,h in faces_list:
            #cv2.rectangle(img_rgb, (x,y), (x+w, y+h), (0,255,0), 2)
            # Step 3: Crop Face
            roi = img[y:y+h, x:x+w]
            cv2.imwrite(f'./crop_data/male/male_{i}.jpg', roi)
            print('Image successfully processed')
    except Exception as err:
        print(f"The error is {err}")
        print(f'Unable to process the image {mpath[i]}')

#Repeat the same for loop for Female Images.