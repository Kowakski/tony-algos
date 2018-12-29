'''
读取文件的每一行，从每一行获取数据
然后分成自己想要的数据结构
这个脚本用来看看阶数对滤波的影响
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

low_fir_cof  = firwin( 9, 1*2/20 )
fir_x_datas = lfilter(low_fir_cof, 1, x_datas)

low_fir_cof2  = firwin( 129, 1*2/20 )
fir_x_datas2 = lfilter(low_fir_cof2, 1, x_datas)

t = [x for x in range(0, len(x_datas))]    #生成时间轴，每0.005s采一个点，时间就是0.005的倍数
t = [time_inter*x for x in t]
fig = plt.figure()
ax1 = fig.add_subplot(111)



ax1.plot(t, x_datas, 'b')
ax1.plot(t, fir_x_datas2, 'g')     #129
ax1.plot(t, fir_x_datas, 'r')      #9
plt.show()