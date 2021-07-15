
# Prepeare data file
import csv
import cv2
import numpy as np

#Function to load data
def load_data(datafolder):
    
    csvfilename=datafolder+'/driving_log.csv'
    imgfolder=datafolder+'/IMG'
    
    lines=[]
    
    #Raed csv file and append each line to list
    with open(csvfilename) as csvfile:
        reader=csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return   lines      
    
            
            
Samples=[]

"""Samples_part1 = load_data('./Data')
Samples.extend(Samples_part1)

Samples_part2 = load_data('./Data2')
Samples.extend(Samples_part2)

Samples_part3  = load_data('./Data3')
Samples.extend(Samples_part3)

Samples_part3 = load_data('./Data4')
Samples.extend(Samples_part3)

Samples_part4 = load_data('./Data5')
Samples.extend(Samples_part4)"""

Samples_part5 = load_data('./data')
Samples.extend(Samples_part5)


#import keras lbrary
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def training_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0- 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(1))
    
    return model
    

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(Samples, test_size=0.2)

import sklearn
import math

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
               if(batch_sample[3]=='steering'):
                continue
               else:
                sttervalue=float(batch_sample[3])
                steer_correction=0.2
                
                for i in range(3):
                    imgpath=batch_sample[i]
                    imgpath_split=imgpath.split('/')
                    
                    img_name=imgpath.split('/')[-1]
                    imgfolder=imgpath.split('/')[-2]
                   
                    if(len(imgpath_split)<3):
                        parentfolder='data'
                    else:
                        parentfolder=imgpath.split('/')[-3]
                    
                    #print(('./'+parentfolder+'/'+imgfolder+'/'+img_name))
                    img = cv2.imread(('./'+parentfolder+'/IMG/'+img_name))
                    #print(img)
                    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    #print(img_rgb.shape)
                    images.append(img_rgb)
                    if i==0 :
                        carsteer=sttervalue
                    else:
                        carsteer=(sttervalue - ((-1**i)*steer_correction))

                    angles.append(carsteer)

                    #flip images
                    #print(img_rgb.shape)
                    images.append(cv2.flip(img_rgb,1))
                    angles.append(carsteer*-1)
                    
                    #Britness change
                    img_hsv=cv2.cvtColor(np.uint8(img_rgb),cv2.COLOR_RGB2HSV)
                    img_hsv[:,:,2] = img_hsv[:,:,2] * np.random.uniform(0.2,1)
                    img_rgb1=cv2.cvtColor(np.uint8(img_hsv),cv2.COLOR_HSV2RGB)
                    
                    #print(img_rgb1.shape)
                    images.append(img_rgb1)
                    angles.append(carsteer)
                    
        # trim image to only see section with road
        X_train = np.array(images)
        y_train = np.array(angles)
        yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# define the network model
model = training_model()
model.compile(loss='mse', optimizer='adam')

trained=model.fit_generator(train_generator, 
                            steps_per_epoch=math.ceil(len(train_samples)*9/batch_size), 
                            validation_data=validation_generator, 
                            validation_steps=math.ceil(len(validation_samples)*9/batch_size), 
                            epochs=5, verbose=1)

model.save('model.h5')

