
# coding: utf-8

# In[2]:

import csv
import cv2
import numpy as np
import sklearn
import os
import matplotlib.image as impimg
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Cropping2D
from sklearn.model_selection import train_test_split

row = 160
col = 320
ch = 3
correction = 0.2

lines = []
with open('data/driving_log.csv') as csvfile:
    next(csvfile)
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


# In[3]:

def augment(samples):
    a_images,a_angles = ([],[])
    for sample in samples:
        center = 'data/IMG/' + sample[0].split('/')[-1]
        right  = 'data/IMG/' + sample[2].split('/')[-1]
        left   = 'data/IMG/' + sample[1].split('/')[-1]
        
        image_center = impimg.imread(center)
        image_right  = impimg.imread(right)
        image_left   = impimg.imread(left)
        
        angle_center = float(sample[3])
        angle_left  = angle_center + correction 
        angle_right = angle_center - correction
        
        a_images.append(image_center)
        a_angles.append(angle_center)
        a_images.append(cv2.flip(image_center,1))
        a_angles.append(angle_center*(-1))
        
        a_images.append(image_left)
        a_angles.append(angle_left)
        a_images.append(cv2.flip(image_left,1))
        a_angles.append(angle_left*(-1))
        
        a_images.append(image_right)
        a_angles.append(angle_right)
        a_images.append(cv2.flip(image_right,1))
        a_angles.append(angle_right*(-1))
        
    return shuffle(a_images,a_angles)

def generator(samples,measurement, batch_size):
    num_samples = len(samples)
    shuffle(samples,measurement)
    while 1:                            
        for offset in range(0, num_samples, batch_size):
            #X_data,y_data = ([],[])
            X_data = samples[offset:offset+batch_size]
            y_data = measurement[offset:offset+batch_size]
            yield shuffle(np.array(X_data),np.array(y_data))


# In[4]:

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

X_train,y_train = augment(train_samples)
X_valid,y_valid = augment(validation_samples)


# In[5]:

def nvidia_arch():
   
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x/255 - 0.5))
    
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid"))
    model.add(Activation('relu'))
  
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid"))
    model.add(Activation('relu'))
    
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid"))
    model.add(Activation('relu'))    
    
    model.add(Convolution2D(64, 3, 3, border_mode="valid"))
    model.add(Activation('relu'))
    
    model.add(Flatten())

    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(10))
    model.add(Activation('relu'))
        
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse") 
    return model


# In[6]:

batch_size = 16
epochs = 4

train_generator = generator(X_train,y_train, batch_size)
validation_generator = generator(X_valid,y_valid, batch_size)

model = nvidia_arch()
model.fit_generator(
    train_generator,
    samples_per_epoch=len(X_train)/batch_size, nb_epoch=epochs,
    validation_data=validation_generator,
    nb_val_samples=len(X_valid)/batch_size
    )

model.save('model.h5')


# In[ ]:

model.summary()

