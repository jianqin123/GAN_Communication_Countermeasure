import os
import scipy.io as sio
import numpy as np
import pandas
import matplotlib
# matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import  random
import  cmath
from scipy.fftpack import fft,fft2,fftn,fftshift
from mpl_toolkits.mplot3d import Axes3D

'''
采用通信中的传统提取特征谱的方式，识别信号的调制类别
'''



# 计算相关函数及其谱密度
def selfCor(x, a):
    length = len(x)
    r = []
    for dis in range(length):
        temp = 0
        for t in range(length - dis):
            temp += x[t] * x[t + dis] * cmath.exp(complex(0, -2 * cmath.pi * a * t/length))
        r.append(temp)
    r=np.array(r)/length

    return  r

def selfCor(x):
    length=len(x)
    z=np.zeros((length,length))
    for i in range(length):
        for j in range(length):
            z[i,j]=x[i]*x[i+j] if i+j<length else 0
    s=fft2(z)
    s_=fftshift(s)
    return abs(s_)

def densiSpectrum(X,fs,save_to):
    length=len(X)
    signal=fftshift(fft(X))
    Y=np.zeros((length,length))
    n=0
    for al in np.arange(int(-length/2),0+1,1):
        Y[n,np.arange(-al+1,length+al,1)]=abs(signal[1:length+2*al]*np.conj(signal[-2*al+1:length]))/length
        n+=1
    Y[n:length,:]=np.conj(Y[np.arange(n-2,1-1,-1),:])


    XX = (np.array(range(length))-int(length/2))*fs/length
    YY= np.arange(-fs,fs,fs/(length/2))
    # XX, YY = np.meshgrid(XX, YY)
    #
    # print(Y[0])
    #
    # #绘制图片
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(XX, YY, abs(Y))
    ax.set_xlabel('f/(hz)')
    ax.set_ylabel('alpha/(hz)')
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()
    plt.savefig(save_to)
    plt.show()
    fig.clear()
    f_0=Y[int(length/2)+1]
    plt.plot(YY,f_0)
    print(abs(YY[np.where(f_0==np.max(f_0))[0][0]]))
    plt.show()
    fig.clear()














# 计算alpha 和f两个维度上面的谱密度函数
def densi_spectrum(x):
    densi = []
    for i in range(len(x)):
        densi.append(selfCor(x, i))
        print(len(densi), densi[0].shape)
    return np.array(densi)




# 绘制三维图
def plot_3D(s,save_to):
    fig = plt.figure()
    ax = Axes3D(fig)
    length = s.shape[0]
    X =np.array( range(length))-int(length/2)
    Y =np.array( range(length))-int(length/2)
    X, Y = np.meshgrid(X, Y)
    surf=ax.plot_surface(X, Y, s, rstride=1, cstride=1, cmap='rainbow')
    ax.set_xlabel('f/(hz)')
    ax.set_ylabel('alpha/(hz)')
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()
    plt.savefig(save_to)
    fig.clear()


type='AM'
path=os.path.join('..','signal_data',type)
# mat=sio.loadmat(path)
# am1=mat['s']
# signal1=am1[0][:128]
# x=np.arange(0,1,1/2000)
# y=np.sin(2*np.pi*10*x) + np.sin(2*np.pi*30*x)
# carries=np.sin(2*np.pi*50*x)
# signal1=y*carries
X=np.arange(0,1,1/2000)
Y=np.sin(2*20*np.pi*X)
Carr=np.sin(2*500*np.pi*X)
signal1=Y *Carr
# r_0=selfCor(signal1,0)
# plt.plot(range(len(signal1)),abs(r_0))
# plt.show()
densiSpectrum(signal1,2000,path)

# s=selfCor(signal1)
# print(s.shape)
# argmax=np.where(s==np.max(s))
# print(argmax)
# plot_3D(s,path)


