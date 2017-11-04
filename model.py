import os
import csv
import cv2
import numpy as np
import math
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        # shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                 for i in range(3):
                    source_name = batch_sample[i]
                    name = 'data/IMG/'+source_name.split('/')[-1] #for sample data
                    # filename = source_name[62:] #for recorded training data
                    # name = 'data/IMG/' + filename #for recorded training data
                    cam_image = cv2.imread(name)
                    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)
                    images.append(cam_image)
                    images.append(cv2.flip(cam_image,1))
                 center_angle = float(batch_sample[3])
                 # correction = 0.2 #for sample data
                 correction = 0.2 #for recorded training data
                 angles.append(center_angle)
                 angles.append(-1.0*center_angle)
                 angles.append(center_angle+correction)
                 angles.append(-1.0*(center_angle+correction))
                 angles.append(center_angle-correction)
                 angles.append(-1.0*(center_angle-correction))
                             
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# print('# of training samples: ', str(len(train_samples)))
# print('# of validation samples: ', str(len(validation_samples)))
# for i in range(0,math.ceil(len(train_samples)/32)):
#     print('Batch ' + str(i))
#     test_x,test_y  = (next(train_generator))
#     print(test_x.shape)
#     print(test_y.shape)
# exit()

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(16, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model_history = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6, validation_data=validation_generator,nb_val_samples=len(validation_samples),nb_epoch=5)

model.save('model_lenetmod_e5_lr_2_aug_crop_st_gen.h5')

# import matplotlib.pyplot as plt

### plot the training and validation loss for each epoch
# plt.plot(model_history.history['loss'])
# plt.plot(model_history.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()