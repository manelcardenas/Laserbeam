import os
import cv2

mask_directory = '/mnt/home.stud/cardejoa/Desktop/bcn/data_vis/Masks_png/'
mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

masks = os.listdir(mask_directory)
for i, mask_name in enumerate(masks):
    if (mask_name.split('.')[1] == 'png'):
        png_img = cv2.imread(mask_directory+mask_name, 0)
        print(mask_directory+mask_name)
        cv2.imwrite('/mnt/home.stud/cardejoa/Desktop/bcn/data/Masks/mask_name.jpg', png_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        #print(mask_directory+mask_name)
        
