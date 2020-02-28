import pickle
import matplotlib.pyplot as plt
with open("D:/history_0704.txt","rb") as f:
    records = pickle.load(f)
fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
loss,acc,val_loss,val_acc = records
iters = range(len(loss['epoch']))
#iters = iters[:80000:700]
ax2.plot(iters,acc['epoch'],'r',label='train acc')
ax1.plot(iters,loss['epoch'],'g',label='train loss')
#ax2.grid(True)#设置网格形式
ax2.set_xlabel('batch')
ax2.set_ylabel('acc')#给x，y轴加注释
ax1.set_ylabel('loss')
ax1.grid()
#fig.grid()
fig.legend(loc="best", bbox_to_anchor=(0.11,1),bbox_transform=ax1.transAxes)#设置图例显示位置
plt.show()
