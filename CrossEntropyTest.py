import tensorflow as tf

labels = [[0,0,1],[0,1,0]]   #标签
logits = [[2,0.5,6], [0.1,0,3]] #样本，第一个样本和标签是相符合的，第二个不符合

logits_scaled = tf.nn.softmax(logits)  #softmax 把样本归一化了
#scaled = [[ 0.01791432  0.00399722  0.97808844]
#          [ 0.04980332  0.04506391  0.90513283]]

logits_scaled2 = tf.nn.softmax(logits_scaled) #再用softmax归一化一次，没有第一次明显了
# scaled2 = [[ 0.21747023  0.21446465  0.56806517]
#            [ 0.2300214   0.22893383  0.54104471]]

result1 = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits )
# rel1 = [ 0.02215516  3.09967351]

result2 = tf.nn.softmax_cross_entropy_with_logits( labels = labels, logits = logits_scaled ) #这个算出来的交叉熵还没有前面一个明显了，就是softmax之后的不要用这个来算交叉熵了
# rel2 = [ 0.56551915  1.47432232]

result3 = -tf.reduce_sum(labels*tf.log(logits_scaled),1)  #自己定义一个算交叉熵的，这个算出来的交叉熵比较理想
# rel3 =  [ 0.02215518  3.09967351]

with tf.Session() as sess:
    print('scaled  =', sess.run(logits_scaled))
    print('scaled2 =', sess.run(logits_scaled2))
    print('rel1    =', sess.run(result1),'\n')
    print('rel2    =', sess.run(result2),'\n')
    print('rel3    = ', sess.run(result3))