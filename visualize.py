import keras
import numpy as np
from keras.models import Model,load_model
import cv2
import pdb
layer_list = [0,1,4,5,8,9,12,13,16,17,20,21] #由断点调试观察得出需要截取的网络层的索引
real_id = [1,1,2,2,3,3,4,4,5,5,6,6]
model = load_model('D:/cifar/my_model_074.h5')#加载模型
img = cv2.imread('D:/cifar/21.jpg')
img = img.astype('float32')
img = img/255
img = cv2.resize(img,(224,224))
img = img.reshape(-1,224,224,3)
is_conv=True
for layer_id in layer_list:
    #layer_index = list(range(len(layer_list)))
    layer_model = Model(inputs=model.input,output=model.layers[layer_id].output) #截取到模型的某一层
    feature = layer_model.predict(img)
    d_1,d_2,d_3,d_4 = feature.shape
    feature = feature.reshape(d_2,d_3,d_4)
    if is_conv:
        layer_name = 'conv'
        is_conv = False
    else:
        layer_name = 'relu'
        is_conv = True
    layer_index = real_id[layer_list.index(layer_id)]
    for i in range(d_4):
        im = feature[:,:,i]
        pdb.set_trace()
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        im = cv2.resize(im,(112,112))
        #if layer_name == 'relu':
            #layer_id-=1
        cv2.imwrite('D:/feature_maps/{}{}_{}.png'.format(layer_name,layer_index,i),im)
print('done')
#cv2.imshow('Window',feature[:,:,40])fea
#cv2.waitKey()
