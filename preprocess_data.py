import numpy as np
import os
import pickle
import pdb
import cv2
#train_dir = "D:/cifar/train/"

def load_data(train_dir):
    total =  0
    for train_batch in os.listdir(train_dir):
        #train_data = np.empty((10000,3,32,32),dtype="int32")
        batch_path = os.path.join(train_dir,train_batch)
    
    
        with open(batch_path,'rb') as f:
            data_dict = pickle.load(f,encoing="bytes")
            print(data_dict.keys())
            pdb.set_trace()
            count = 0
            for img_data in data_dict[b'data']:
                #print(img_data)
                #pdb.set_trace()
                r_channel = np.empty((32,32))
                g_channel = np.empty((32,32))
                b_channel = np.empty((32,32))
                r_channel = img_data[0:1024].reshape(32,32)
                g_channel = img_data[1024:2048].reshape(32,32)
                b_channel = img_data[2048:3072].reshape(32,32)
                image = cv2.resize(cv2.merge([r_channel,g_channel,b_channel]),(224,224))
                cv2.imwrite("D:/cifar/test/img"+str(total)+'_'+str(data_dict[b'labels'][count])+'.png',image)
                print("saving{} images".format(total))
                if total>= 9999:
                    return
                #cv2.imshow('lll',image)
                #cv2.waitKey(0)
                #print(train_data[count])
                #pdb.set_trace()
                count+=1
                total+=1
if __name__=="__main__":
    train_dir = "D:/cifar/test_batch/"
    load_data(train_dir)
#print("save done...")
