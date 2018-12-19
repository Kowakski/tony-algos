'''
1. 随机选取K个点作为中心点
2. 计算所有点离所有中心点的距离
3. 所有的点自己看离哪个中心点近，就把自己归为那一类
4. 重新计算中心点，所有属于那一类的点做平均值
5. 重复2~4步骤，直到达到MAX_ITERS次
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets.samples_generator import make_circles



K = 9 # 类别数目
MAX_ITERS = 1000 # 最大迭代次数
N = 200 # 样本点数目
data_file_path = './data_cache_file.pkl'

if not os.path.exists( data_file_path ):
    exit(  )

with open( data_file_path, 'rb' ) as f:
    data = pickle.load( f )
    data = np.array( data )
    N = len( data )

#随机初始化9个中心点
centers = [ [36.0, 19.0], [40.0, 61.0], [49.0, 89.0], [92.0, 63.0], [129.0, 105.0], [150.0, 198.0], [177.0, 91.0], [248.0, 226.0], [397.0, 264.0] ] # 簇中心

# 生成人工数据集
#data, features = make_circles(n_samples=200, shuffle=True, noise=0.1, factor=0.4)
# data, features = make_blobs(n_samples=N, centers=centers, n_features = 2, cluster_std=0.8, shuffle=False, random_state=42)
print(data)
# print(features)

# 计算类内平均值函数
def clusterMean(data, id, num):
    total = tf.unsorted_segment_sum(data, id, num)     # 第一个参数是tensor，第二个参数是簇标签，第三个是簇数目
    count = tf.unsorted_segment_sum(tf.ones_like(data), id, num)
    return total/count

# 构建graph
points = tf.Variable(data)
cluster = tf.Variable(tf.zeros([N], dtype=tf.int64))
centers = tf.Variable(tf.slice(points.initialized_value(), [0, 0], [K, 2]))# [第一步] 将原始数据前k个点当做初始中心

repCenters = tf.reshape(tf.tile(centers, [N, 1]), [N, K, 2]) # 复制操作，便于矩阵批量计算距离
repPoints = tf.reshape(tf.tile(points, [1, K]), [N, K, 2])

sumSqure = tf.reduce_sum(tf.square(repCenters-repPoints), reduction_indices=2) # [第二步] 计算距离
bestCenter = tf.argmin(sumSqure, axis=1)  # [第三步] 寻找最近的簇中心

change = tf.reduce_any(tf.not_equal(bestCenter, cluster)) # 检测簇中心是否还在变化
means = clusterMean(points, bestCenter, K)  #[第四步] 计算簇内均值

# 将粗内均值变成新的簇中心，同时分类结果也要更新
with tf.control_dependencies([change]):
    update = tf.group(centers.assign(means),cluster.assign(bestCenter)) # 复制函数

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    changed = True
    iterNum = 0
    while changed and iterNum < MAX_ITERS:    #这个例子里面循环了8次，changed就变成false了
        iterNum += 1
        # 运行graph
        [changed, _] = sess.run([change, update])
        [centersArr, clusterArr] = sess.run([centers, cluster])
        print(clusterArr)     #属于哪一类
        print(centersArr)     #中心点坐标
        print( "iterNum is {} {} {}".format( iterNum, MAX_ITERS, changed ) )
        # 显示图像
        if (iterNum%20 == 0) or not ( changed ):
            fig, ax = plt.subplots()
            ax.scatter(data.transpose()[0], data.transpose()[1], marker='o', s=100, c=clusterArr)
            plt.plot()
            plt.show()




#最后得到的anchorbox是：
#[[48.10839161, 54.1451049], [106.54423592, 134.44235925], [110.96802326, 87.44476744], [153.25862069, 204.18226601], [171.10289389, 130.66559486], [205.24081633, 257.34285714], [250.75, 194.33333333], [307.76, 332.9], [404.57894737, 169.49122807]]
#原始图像的尺寸是1280*720，darknet-53的输入是416*416，所以这个框要按比例修改
#original_box = [[48.10839161, 54.1451049], [106.54423592, 134.44235925], [110.96802326, 87.44476744], [153.25862069, 204.18226601], [171.10289389, 130.66559486], [205.24081633, 257.34285714], [250.75, 194.33333333], [307.76, 332.9], [404.57894737, 169.49122807]]
#ration = [416/1280, 416/720]
#mp.multiply( a,b )得到结果
#array([[ 15.63522727,  31.28383839],
      # [ 34.62687667,  77.67780757],
      # [ 36.06460756,  50.52364341],
      # [ 49.80905172, 117.97197592],
      # [ 55.60844051,  75.49567703],
      # [ 66.70326531, 148.68698413],
      # [ 81.49375   , 112.28148148],
      # [100.022     , 192.34222222],
      # [131.4881579 ,  97.92826511]])