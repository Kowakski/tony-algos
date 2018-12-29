'''
读取文件的每一行，从每一行获取数据
然后分成自己想要的数据结构
这个脚本把xyz提取出来，然后sqrt(x平方+y平方+z平方)，把这个波形画出来看看是个什么样子的
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

low_fir_cof  = firwin(29, [(0.2*2/20), (5*2/20)], pass_zero = False)
fir_x_datas = lfilter(low_fir_cof, 1, x_datas)
fir_y_datas = lfilter(low_fir_cof, 1, y_datas)
fir_z_datas = lfilter(low_fir_cof, 1, z_datas)

t = [x for x in range(0, len(x_datas))]    #生成时间轴，每0.005s采一个点，时间就是0.005的倍数
t = [time_inter*x for x in t]
fig = plt.figure()
ax1 = fig.add_subplot(111)

square_signal = np.sqrt(np.square(x_datas) + np.square(y_datas) + np.square(z_datas) )
fir_square_signal = np.sqrt(np.square(fir_x_datas) + np.square(fir_y_datas) + np.square(fir_z_datas) )

ax1.plot(t, square_signal, 'r')
ax1.plot(t, fir_square_signal, 'g')

plt.show()