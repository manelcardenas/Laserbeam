import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

directory = '/mnt/home.stud/cardejoa/Desktop/bcn/'


def histo_equalized_images(i,eq_image):
    cv2.imwrite(directory+'results/equalized_images/image_equalized_{}.png'.format(i),eq_image)
   

def images_to_model(X_test,y_test):
#Sanity check, images used to train the model after pre-processing
#image_number = random.randint(0, len(X_train))  #for testing with just one random image
    
    #print('imagen:',X_train[image_number])
    #print('mask:',y_train[image_number])

    for i in range(len(X_test)):
        ##scale_img = X_test[i]
        scale_img = 255 * X_test[i]
        img = scale_img.astype(np.uint8)
        cv2.imwrite(directory+'results/Sanity_check/check_image_{}.jpg'.format(i),img)
        scale_mask = 255 * y_test[i]
        mask = scale_mask.astype(np.uint8)
        cv2.imwrite(directory+'results/Sanity_check/check_mask_{}.png'.format(i),mask)

def loss_error_function(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(directory+'results/loss.jpg')
    plt.close()

def metrics_function(history):
    ##acc = history.history['accuracy']
    ##val_acc = history.history['val_accuracy']
    IoU = history.history['IoU']
    val_IoU = history.history['val_IoU']
    epochs = range(1, len(IoU) + 1)
    plt.plot(epochs, IoU, 'y', label='Training IoU')
    plt.plot(epochs, val_IoU, 'r', label='Validation IoU')
    plt.title('Training and validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()
    plt.show()
    plt.savefig(directory+'results/IoU.jpg')
    plt.close()


