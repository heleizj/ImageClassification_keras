from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import math
import pdb

img_size = 224
data_path = "D:/cifar/img/"
test_path = "D:/cifar/test/"
weight_decay = 0.0005
learning_rate = 0.1
lr_decay = 1e-6
lr_drop = 20

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        #创建一个图
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')#plt.plot(x,y)，这个将数据画成曲线
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)#设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')#给x，y轴加注释
        plt.legend(loc="upper right")#设置图例显示位置
        plt.show()
        
history = LossHistory()

def data_generator(path,batch_size):
    file_list = [os.path.join(path,img_name) for img_name in os.listdir(path)]
    random.shuffle(file_list)
    batch_num = math.ceil(len(file_list)/batch_size)
    while True:
        for i in range(batch_num):
            if i == batch_num - 1:
                batch_list = file_list[batch_size*i:]
            else:
                batch_list = file_list[batch_size*i:batch_size*i+batch_size]
            train_x = np.empty((batch_size,img_size,img_size,3),dtype="int32")
            train_y = np.empty((batch_size,),dtype="int32")
            count = 0
            for img_path in batch_list:
                train_x[count] = cv2.imread(img_path)
                train_y[count] = int(img_path.split('_')[-1].split('.')[0])
                count+=1
            train_data = train_x.astype('float32')
            train_data = train_data/255
            train_label = keras.utils.to_categorical(train_y,10)
            #pdb.set_trace()
            yield (train_data,train_label)
            
def validate_generator(test_path):
    file_list = [os.path.join(test_path,img_name) for img_name in os.listdir(test_path)]
    file_list = file_list[:100]
    test_x = np.empty((len(file_list),img_size,img_size,3))
    test_y = np.empty((len(file_list),))
    count = 0
    for img_path in file_list:
        test_x[count] = cv2.imread(img_path)
        test_y[count] = img_path.split('_')[-1].split('.')[0]
        count+=1
    test_x = test_x.astype('float32')
    test_x = test_x/255
    test_y = keras.utils.to_categorical(test_y,10)
    return(test_x,test_y)

#def lr_scheduler(epoch):
    #return learning_rate * (0.5 ** (epoch // lr_drop))
#reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

(test_data,test_label) = validate_generator(test_path)

model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same',
                input_shape=(224,224,3),kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))
#sgd = optimizers.SGD(lr=learning_rate,momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit_generator(generator=data_generator(data_path,8),
                              epochs=60,
                              steps_per_epoch=2500,
                              validation_data=(test_data,test_label),
                              callbacks=[history])
#data_dict = history.history
records = [history.losses,history.accuracy,history.val_loss,history.val_acc]
with open("D:/history_0704.txt","wb") as f:
    pickle.dump(records,f)
    print("save the history data done:)")
model.save("D:/cifar/my_model_074.h5")
