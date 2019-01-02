'''
1.打开文件获取文件中的数据，用来模拟从acc  sensor获取的数据
2.数据进行分割
3.提取特征
4.获取训练好的模型
5.对特征进行预测
6.显示预测结果
'''
import sys
import pdb
import math
import numpy as np
from os import path
from svmutil import *

model_path = './raw_trained.model'
data_file = '../../WISDM_ar_v1.1_raw.txt'

if not path.exists( model_path ):   #文件不存在，先训练模型
	print("no model file, train your data first")
	exit()

model = svm_load_model( model_path )

segnum=100     #每次切割多少个数据点组成一组来提取特征

if not path.exists( data_file ):
	print("no data file")

'''
最大值，最小值，均值，标准差
'''
FEATURE = ("mean", "max", "min", "std")
def get_features(x,y,z,timestamp):
	# pdb.set_trace()
	# gravity = np.sqrt(np.square(x)+np.square(y)+np.square(z))
	# pdb.set_trace()
	axis = list(zip(x,y,z))
	gravity = [math.sqrt(math.pow(sampnode[0], 2)+math.pow(sampnode[1], 2)+math.pow(sampnode[2], 2)) for sampnode in axis]
	# print("x is {}".format(x))
	# print("y is {}".format(y))
	# print("z is {}".format(z))
	# print("gravity is {}".format(gravity))
	mean    = np.mean(gravity)
	maxx    = np.max(gravity)
	minx    = np.min(gravity)
	std     = np.std(gravity)
	features = {FEATURE.index('mean'):mean, FEATURE.index("max"):maxx, FEATURE.index('min'):minx, FEATURE.index("std"):std}
	return features

def prepare_next_feature():
	x=[]
	y=[]
	z=[]
	timestamp = []
	count = 0

f  = open(data_file)

x=[]
y=[]
z=[]
timestamp = []
count = 0
line = f.readline( )
prepare_next_feature( )

while line:
	count += 1
	line = line.strip().lstrip().rstrip(';')        #去掉末尾的换行符号什么的
	line = line.split(',')     #把数据分开成list，字符串形式的
	# print(line)
	timestamp.append(int(line[2]))
	x.append(float(line[3]))
	y.append(float(line[4]))     #获取y轴的数据
	z.append(float(line[5]))
	line = f.readline()
	if count%segnum == 0:
		# pdb.set_trace()
		feature_vector = get_features(x, y, z, timestamp)
		print(feature_vector)
		p_labs, p_acc, p_vals = svm_predict([], [feature_vector], model)
		print(p_labs, end=' ')
		prepare_next_feature( )

f.close()