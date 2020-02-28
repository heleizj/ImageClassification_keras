import keras
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D,BatchNormalization
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import pdb
img_size = 224
data_path = "D:/cifar/img/"
test_path = "D:/cifar/test/"

def load_data(data_path,test_path):
    imgs = os.listdir(data_path)
    test_imgs = os.listdir(test_path)
    train_data = np.empty((3500,img_size,img_size,3),dtype="int32")
    train_label = np.empty((3500,),dtype="int32")
    test_data = np.empty((1000,img_size,img_size,3),dtype="int32")
    test_label = np.empty((1000,),dtype="int32")
    i=0
    for img_name in imgs:
        img_path = os.path.join(data_path,img_name)
        train_data[i] = cv2.imread(img_path)
        train_label[i] = int(img_name.split('_')[-1].split('.')[0])
        #pdb.set_trace()
        if i==3499:
            break
        i+=1
    #pdb.set_trace()
    i=0
    for test_name in test_imgs:
        img_path = os.path.join(test_path,test_name)
        test_data[i] = cv2.imread(img_path)
        test_label[i] = int(test_name.split('_')[-1].split('.')[0])
        if i==999:
            break
        i+=1
    return train_data,train_label,test_data,test_label

train_data, train_label, test_data, test_label = load_data(data_path,test_path)
train_data,test_data = train_data.astype('float32'),test_data.astype('float32')
train_data,test_data = train_data/255,test_data/255

train_label = keras.utils.to_categorical(train_label,10)
test_label = keras.utils.to_categorical(test_label,10)

#AlexNet
model = Sequential()

#conv1
model.add(Conv2D(filters=96,kernel_size=(11,11),
                 strides=(4,4),padding='valid',
                 input_shape=(img_size,img_size,3),
                 activation='relu'))

model.add(BatchNormalization())

#pooling
model.add(MaxPooling2D(pool_size=(3,3),
                      strides=(2,2),
                      padding='valid'))

#conv2
model.add(Conv2D(filters=256,kernel_size=(5,5),
                 strides=(1,1),padding='same',
                 activation='relu'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3),
                      strides=(2,2),
                      padding='valid'))
#conv3
model.add(Conv2D(filters=384,kernel_size=(3,3),
                 strides=(1,1),padding='same',
                 activation='relu'))
#conv4
model.add(Conv2D(filters=384,kernel_size=(3,3),
                 strides=(1,1),padding='same',
                 activation='relu'))
#conv5
model.add(Conv2D(filters=256,kernel_size=(3,3),
                 strides=(1,1),padding='same',
                 activation='relu'))
#pooling
model.add(MaxPooling2D(pool_size=(3,3),
                       strides=(2,2),padding='valid'))
#fc
model.add(Flatten())
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.5))

#output Layer
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(train_data,train_label,
          batch_size = 16,
          epochs = 120,
          validation_split=0.1,
          shuffle=True)
scores = model.evaluate(train_data,train_label,verbose=1)
#返回的是 损失值和选定的指标值（这里是精度,accuracy）
print(scores)
scores = model.evaluate(test_data,test_label,verbose=1)
print(scores)
model.save("D:/cifar/my_model_416.h5")
