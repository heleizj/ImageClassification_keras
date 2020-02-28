import matplotlib.pyplot as plt
import pickle
import numpy as np
import pdb
with open("C:/Users/z/Desktop/history_vgg_1.txt","rb") as f:
    records = pickle.load(f)
loss,acc,val_loss,val_acc = records
fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
x = np.arange(1,51)
x[0]=1
ax1.plot(x,loss['epoch'][:50],color='c',marker='*',linewidth=1.8,label="train_loss")
ax1.plot(x,val_loss['epoch'][:50],color='orange',marker='o',linewidth=1.8,label="val_loss")
#ax2.plot(history_dict['acc'][:58],color = 'green',label="train_acc")
ax2.plot(x,val_acc['epoch'][:50],color='blue',marker='v',linewidth=1.8,label="val_acc")
#pdb.set_trace()
#plt.xticks(new_xticks)
ax1.set_xlabel("Epoch",fontsize=15)
ax1.set_ylabel("Loss",fontsize=15)
ax2.set_ylabel("Accuracy",fontsize=15)
#ax1.set_title("CNN acc-loss")
#ax1.grid()
fig.legend(loc=1, bbox_to_anchor=(0.9,0.5),fontsize='medium',bbox_transform=ax1.transAxes)
#bbox_to_anchor(num1, num2),  bbox_to_anchor被赋予的二元组中，第一个数值用于控制legend的左右移动，
#值越大越向右边移动，第二个数值用于控制legend的上下移动，值越大，越向上移动。
#ax2.legend(loc='best')
#plt.legend(['losss','val-loss','acc','val-acc'],loc='upper left')
plt.show()
'''
plt.plot(history_dict['acc'])
plt.plot(history_dict['val_acc'])
plt.plot(history_dict['loss'])
plt.plot(history_dict['val_loss'])
plt.title("model acc-loss")
plt.xlabel("epoch")
plt.ylabel("acc-loss")
plt.legend(['train-acc','val-acc','train-loss','val-loss'],loc='upper left')
plt.show()
'''


