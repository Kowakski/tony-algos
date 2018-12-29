'''
读取文件的每一行，从每一行获取数据
然后分成自己想要的数据结构
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
from scipy.signal import firwin, lfilter

import pdb

file = open('./walking1.txt')

x_datas = []
y_datas = []
z_datas = []

for line in file.readlines():
	curLine = line.strip()    #strip()是去掉行后面的换行符号，split是按照都好分开
	curLine = curLine[:-1].split(',')    #行最后的分号去掉

	x = float(curLine[3])
	y = float(curLine[4])
	z = float(curLine[5])
	x_datas.append(x)
	y_datas.append(y)
	z_datas.append(z)

print("Read line from file complete")

time_inter = 5e-3       #0.005S采集一个点 1/Fs
Fs = 20           #20Hz

assert len(x_datas ) == len(y_datas) and len(y_datas) == len(z_datas)  #三个轴的数据长度肯定是相等的

t = [x for x in range(0, len(x_datas))]    #生成时间轴
t = [time_inter*x for x in t]
fig = plt.figure()
ax1 = fig.add_subplot(241)

#把波形画出来看一看
ax1.plot(t, x_datas)
ax1.plot(t, y_datas)
ax1.plot(t, z_datas)

#画频率图
ax1 = fig.add_subplot(242)
xx = fft(x_datas)
xx = xx/len(x_datas)     #归一化
xx = xx[range(len(x_datas)//2)]

k = np.arange(len(x_datas))/len(x_datas)
frq = Fs*k          #生成频率轴

# pdb.set_trace()
frq = frq[range(len(x_datas)//2)]       #采样是频率的两倍，变换之后频率是对称的
ax1.plot(frq, abs(xx), 'r')

'''
设计滤波器，过滤掉指定的频率，把信号重新画出来，低通频率是多少，是从上面画图看出来的
只保留1Hz以下的看看
'''
#FIR 1hz的低通滤波器
low_fir_cof = firwin(29, 1*2/20)
fir_signal = lfilter(low_fir_cof, 1, x_datas)
ax2 = fig.add_subplot(243)
ax2.plot(t, fir_signal)


#FIR 1hz~3Hz的带通滤波器
band_fir = firwin(29, [1*2/20, 3*2/20], pass_zero = False)
band_fir_signal = lfilter(band_fir, 1, x_datas)
ax3 = fig.add_subplot(244)
ax3.plot(t, band_fir_signal)

plt.show()