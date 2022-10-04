import cv2

directory = '/mnt/home.stud/cardejoa/Desktop/bcn/results/predicted_images'
images = os.listdir(directory)
images.sort()

for i, image_name in enumerate(images,0):
    if (image_name.split('.')[1] == 'jpg'):
        initial_image = cv2.imread(directory+image_name)
        scale_mask = 255 * y_train[i]
        mask = scale_mask.astype(np.uint8)
        cv2.imwrite(directory+'results/Sanity_check/check_mask_{}.png'.format(i),mask)
