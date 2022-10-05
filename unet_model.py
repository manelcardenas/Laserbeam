from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout
#from keras.metrics import MeanIoU
from keras.metrics import binary_crossentropy, accuracy
from keras import backend as K
from metrics.metrics import IoU, dice_score, IoU_loss, dice_score_loss, bce_dice

k = 3 #kernel size
s = 2 #stride

def unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, n_filters, metrics, loss, optimizer):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    #Encoder path
    conv1 = Conv2D(n_filters, (k, k), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv2D(n_filters, (k, k), activation='relu', kernel_initializer='he_normal', padding='same')(conv1)
    pool1 = MaxPooling2D((s, s))(conv1)
    
    conv2 = Conv2D(n_filters*2, (k, k), activation='relu', kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Conv2D(n_filters*2, (k, k), activation='relu', kernel_initializer='he_normal', padding='same')(conv2)
    pool2 = MaxPooling2D((s, s))(conv2)
     
    conv3 = Conv2D(n_filters*4, (k, k), activation='relu', kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(n_filters*4, (k, k), activation='relu', kernel_initializer='he_normal', padding='same')(conv3)
    pool3 = MaxPooling2D((s, s))(conv3)
     
    conv4 = Conv2D(n_filters*8, (k, k), activation='relu', kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(n_filters*8, (k, k), activation='relu', kernel_initializer='he_normal', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)
     
    conv5 = Conv2D(n_filters*16, (k, k), activation='relu', kernel_initializer='he_normal', padding='same')(pool4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(n_filters*16, (k, k), activation='relu', kernel_initializer='he_normal', padding='same')(conv5)
    
    #Decoder path 
    up6 = Conv2DTranspose(n_filters*8, (s, s), strides=(s, s), padding='same')(conv5)
    up6 = concatenate([up6, conv4])
    conv6 = Conv2D(n_filters*8, (k, k), activation='relu', kernel_initializer='he_normal', padding='same')(up6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(n_filters*8, (k, k), activation='relu', kernel_initializer='he_normal', padding='same')(conv6)
     
    up7 = Conv2DTranspose(n_filters*4, (s, s), strides=(s, s), padding='same')(conv6)
    up7 = concatenate([up7, conv3])
    conv7 = Conv2D(n_filters*4, (k, k), activation='relu', kernel_initializer='he_normal', padding='same')(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(n_filters*4, (k, k), activation='relu', kernel_initializer='he_normal', padding='same')(conv7)
     
    up8 = Conv2DTranspose(n_filters*2, (s, s), strides=(s, s), padding='same')(conv7)
    up8 = concatenate([up8, conv2])
    conv8 = Conv2D(n_filters*2, (k, k), activation='relu', kernel_initializer='he_normal', padding='same')(up8)
    conv8 = Dropout(0.1)(conv8)
    conv8 = Conv2D(n_filters*2, (k, k), activation='relu', kernel_initializer='he_normal', padding='same')(conv8)
     
    up9 = Conv2DTranspose(n_filters, (s, s), strides=(s, s), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(n_filters, (k, k), activation='relu', kernel_initializer='he_normal', padding='same')(up9)
    conv9 = Dropout(0.1)(conv9)
    conv9 = Conv2D(n_filters, (k, k), activation='relu', kernel_initializer='he_normal', padding='same')(conv9)
     
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = metrics)
    model.summary()
    
    return model
