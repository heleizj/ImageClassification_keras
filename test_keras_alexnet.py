import keras
from keras.models import load_model
import cv2
import pickle
import os
import pdb

def load_data(path):
    img = cv2.imread(path)
    img = img.astype('float32')
    img = img/255
    return img.reshape(-1,224,224,3)
        
def get_name(index_num):
    with open("D:/cifar/cifar-10-batches-py/batches.meta","rb") as f:
        data_dict = pickle.load(f,encoding="bytes")
        #pdb.set_trace()
        name_list = data_dict[b'label_names']
        predict_name = str(name_list[index_num]).split('\'')[1]
    return predict_name

def show_picture(img_path):
    model = load_model('D:/cifar/my_model.h5')
    for pic_name in os.listdir(img_path):
        pic_path = os.path.join(img_path,pic_name)
        prediction_array = model.predict(load_data(pic_path))
        #pdb.set_trace()
        prediction = list(prediction_array[0])
        index_ID = prediction.index(max(prediction))
        predict_name = get_name(index_ID)
        img = cv2.imread(pic_path)
        img = cv2.resize(img,(400,400))
        #print(img.shape)
        cv2.putText(img,str(predict_name),(8,20),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow("Window",img)
        key = cv2.waitKey()
        if key != 27:
            continue
        else:
            return

if __name__ =="__main__":
    show_picture("D:/cifar/visualize/")
