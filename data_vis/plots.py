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
    metric = list(history.history.keys())
    metrics = history.history[metric[1]]
    val_metrics = history.history[metric[3]]
    epochs = range(1, len(metrics) + 1)
    plt.plot(epochs, metrics, 'y', label='Training metrics')
    plt.plot(epochs, val_metrics, 'r', label='Validation metrics')
    plt.title('Training and validation metrics')
    plt.xlabel('Epochs')
    plt.ylabel(metric[1])
    plt.legend()
    plt.show()
    plt.savefig(directory+'results/metrics.jpg')
    plt.close()

def predict_images(X_test, y_test, y_pred_threshold):

#test_img_number = random.randint(0, len(y_pred)) for testing with just one random image

    for i in range(len(X_test)):

        test_img = X_test[i]
        ground_truth=y_test[i]
        prediction = y_pred_threshold[i].astype(np.uint8)

 
        scale_final_img = test_img[:,:,:]
        final_test_img = scale_final_img.astype(np.uint8)
        cv2.imwrite(directory+'results/predicted_images/final_test_img_{}.png'.format(i),final_test_img)
   
        ground_truth_in = 255 * ground_truth[:,:,0]
        ground_truth = ground_truth_in.astype(np.uint8)
        cv2.imwrite(directory+'results/predicted_images/ground_truth_{}.png'.format(i),ground_truth)
    
        predict = 255 * prediction
        cv2.imwrite(directory+'results/predicted_images/predict_{}.png'.format(i),predict)
 


