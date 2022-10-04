import tensorflow as tf
import argparse
import torch
import torchvision.transforms as transforms
from unet_model import unet_model   #Use normal unet model
from keras.utils.np_utils import normalize
import os
import cv2
from PIL import Image      #pillow to resize images
import numpy as np
from matplotlib import pyplot as plt
import csv
from csv import writer
from patchify import patchify, unpatchify 
from data_vis.plots import histo_equalized_images, images_to_model, loss_error_function, metrics_function, predict_images
from metrics.metrics import IoU, dice_score, IoU_loss, dice_score_loss, bce_dice	
from sklearn.model_selection import train_test_split
from keras.metrics import accuracy

#Main directories of the project 
image_directory = '/mnt/home.stud/cardejoa/Desktop/bcn/data/fit_model/Images/'   #images used to fit the model, if you use pre-process script, copy the directory where the images are stored   
mask_directory = '/mnt/home.stud/cardejoa/Desktop/bcn/data/fit_model/Masks/'


parser = argparse.ArgumentParser(
     prog='PROG',
     formatter_class=argparse.RawTextHelpFormatter,
     description='The code can be executed without any parameters and it will use the default ones (which are the ones with the best results)')
parser.add_argument('--epochs', type = int, default = 40, help = 'Number of epochs used to train the model. Default: 40')
parser.add_argument('--batch', type = int, default = 32, help = 'Size of the batch. Default: 32')
parser.add_argument('--filters', type = int, default = 16, help = 'Number of filters of the first convolutional layer. Default: 16')
parser.add_argument('--metrics', type = str, help = 'Evaluation metrics that with which the model will be trained. There are 3 options:\n-> IoU for Intersection over Union.\n-> accuracy for the global accuracy.\n-> dice_score for the Dice coefficient')
parser.add_argument('--loss', type = str, help = 'Loss function used to train the model. There are two options:\n-> binary_crossentropy\n-> dice_score_loss')
parser.add_argument('--optimizer', type = str, default = 'adam', help = 'Optimizer used to train the model. Default: Adam')
args = parser.parse_args()

if args.epochs:
    epochs = args.epochs
    print('The number of epochs used will be: ',args.epochs)
else:
    epochs = 40
#puedo poner type = int, nargs = ? , help = 'Number of epochs') para obligar a que solo se ponga 1 valor, pero cuidado con que pasa si no se pone ninguno
if args.batch:
    batch_size = args.batch
    print('The batch size used will be: ',args.batch)
else:
    batch_size = 32
if args.filters:
    n_filters = args.filters
else:
    n_filters = 16
if args.metrics == 'IoU':
    metrics = IoU
    print('The metric used will be: ',args.metrics)
elif args.metrics == 'dice_score':
    metrics = dice_score
    print('The metric used will be: ',args.metrics)
elif args.metrics == 'accuracy':
    metrics = 'accuracy'
    print('The metric used will be: ',args.metrics)
elif args.metrics == None:
    print('The metric used will be IoU, the default one')
    metrics = IoU
else:
    print('The metric is not valid, please try again or use the default')
    exit()

if args.loss == 'binary_crossentropy':
    loss = 'binary_crossentropy'
    print('The loss function used will be: ', args.loss)
elif args.loss == 'dice_score_loss':
    loss = dice_score_loss
    print('The loss function used will be: ', args.loss)
elif args.loss == None:
    print('The loss function used will be binary_crossentropy, the default one')
    loss = 'binary_crossentropy'
else:
    print('The loss function is not valid, please try again or use the default')
    exit()

if args.optimizer:
    optimizer = args.optimizer
else:
    optimizer = 'adam'


SIZE = 1024
image_dataset = []    
mask_dataset = []  

#Availability of GPU/CPU
'''
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print('Please install GPU version of TF')

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
'''

images = os.listdir(image_directory)
images.sort()                                 #So images and masks have the same order to match them
for i, image_name in enumerate(images,0):     
    if (image_name.split('.')[1] == 'jpg'):
        print(image_directory+image_name)
        image = cv2.imread(image_directory+image_name, cv2.IMREAD_COLOR)      #cv2.imread(path, flag); flag=0 -> GRAY_SCALE
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))                          #If images are pre-proceed, already 1024x1024. If not this will resize them 
        image_dataset.append(np.array(image))


masks = os.listdir(mask_directory)
masks.sort()
for i, mask_name in enumerate(masks,0):
    if (mask_name.split('.')[1] == 'png'):
        print(mask_directory+mask_name)
        mask = cv2.imread(mask_directory+mask_name, 0)
        mask = Image.fromarray(mask)
        mask = mask.resize((SIZE,SIZE)) 
        mask_dataset.append(np.array(mask))

#Normalize images
image_dataset = normalize(image_dataset)

#Rescaling of mask of 0 to 1.
mask_dataset = np.expand_dims((mask_dataset),3) /255.



X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.10, random_state = 0)


#Sanity check, images used to train the model after pre-processing
images_to_model(X_test,y_test)
#images_to_model(X_train,y_train)


IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

def get_model():
    return unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, n_filters, metrics, loss, optimizer)

model = get_model()

#callbacks = tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss')                    
history = model.fit(X_train, y_train, 														
                    batch_size = batch_size, 
                    verbose=1, 
                    epochs=epochs, 
                    #callbacks=callbacks, 
                    validation_data=(X_test, y_test), 
                    shuffle=False)

############################################################
#Evaluate the model

score = model.evaluate(X_test, y_test, verbose = 0)
print('Loss = ', score[0])
print('Metric score = ',score[1])
loss_error_function(history)
metrics_function(history)

y_pred=model.predict(X_test)
y_pred_threshold = y_pred > 0.5

#######################################################################
#Prediction of images

predict_images(X_test, y_test, y_pred_threshold)
















