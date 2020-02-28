import matplotlib.pyplot as plt
import pickle
import numpy as np
with open("D:/history_0704.txt","rb") as f:
    history_dict = pickle.load(f)
loss,acc,val_loss,val_acc = history_dict
fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
x = range(60)
ax1.plot(x,loss['epoch'],color='c',marker='*',linewidth=1.8,label="train_loss")
ax1.plot(x,val_loss['epoch'],color='orange',linewidth=1.8,marker='o',label="val_loss")
#ax2.plot(history_dict['acc'][::2],color = 'green',label="train_acc")
ax2.plot(x,val_acc['epoch'],color='blue',linewidth=1.8,marker='v',label="val_acc")
#new_xticks=np.linspace(0,50,6)
#ax1.xticks(new_xticks)
ax1.set_xlabel("epoch",fontsize=15)
ax1.set_ylabel("loss",fontsize=15)
ax2.set_ylabel("accuracy",fontsize=15)
#ax1.set_title("CNN acc-loss")
#ax1.grid()
fig.legend(loc=1, bbox_to_anchor=(0.9,0.5),fontsize='medium',bbox_transform=ax1.transAxes)
#bbox_to_anchor(num1, num2),  bbox_to_anchor被赋予的二元组中，第一个数值用于控制legend的左右移动，
#值越大越向右边移动，第二个数值用于控制legend的上下移动，值越大，越向上移动。
#ax2.legend(loc='best')
#plt.legend(['losss','val-loss','acc','val-acc'],loc='upper left')
plt.show()
