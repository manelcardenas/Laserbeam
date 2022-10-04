import tensorflow as tf
import torch
import torchvision.transforms as transforms
import os
import cv2
from PIL import Image      #pillow to resize images
import numpy as np
from matplotlib import pyplot as plt
from keras.utils.np_utils import normalize
import csv
from csv import writer 
from data_vis.plots import histo_equalized_images, images_to_model	


#Main directories of the project 
image_directory = '/mnt/home.stud/cardejoa/Desktop/bcn/data/raw_data/Images/' #directory where the raw images are stored
mask_directory = '/mnt/home.stud/cardejoa/Desktop/bcn/data/raw_data/Masks/'   
result_directory = '/mnt/home.stud/cardejoa/Desktop/bcn/data/fit_model/'  #directory where the images used to fit the model will be stored



SIZE = 1024
image_dataset = []    


images = os.listdir(image_directory)
images.sort()                                 #So images and masks have the same order to match them
for i, image_name in enumerate(images,0):     
    if (image_name.split('.')[1] == 'jpg'):
        print(image_directory+image_name)
        image = cv2.imread(image_directory+image_name, cv2.IMREAD_COLOR) #cv2.imread(path, flag); flag=0 -> GRAY_SCALE
        img_yuv = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        eq_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        histo_equalized_images(i,eq_image)
        image = Image.fromarray(eq_image)
        transform = transforms.CenterCrop(SIZE)
        central_img_crop = transform(image) 
        image = np.array(central_img_crop)
        cv2.imwrite(result_directory+'Images/img{}.jpg'.format(i),image)

masks = os.listdir(mask_directory)
masks.sort()
for i, mask_name in enumerate(masks,0):
    if (mask_name.split('.')[1] == 'png'):
        print(mask_directory+mask_name)
        mask = cv2.imread(mask_directory+mask_name, 0)
        mask = cv2.bitwise_not(mask)			#mask inversion
        mask = Image.fromarray(mask)
        transform = transforms.CenterCrop(SIZE)
        central_mask_crop = transform(mask) 
        mask = np.array(central_mask_crop)
        cv2.imwrite(result_directory+'Masks/mask{}.png'.format(i),mask)

      
print('Done!') 