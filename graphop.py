#Is reading book pracitce codes:
#chaptor 4.4

import numpy as np
import tensorflow as tf
c = tf.constant(0.0)    #there is a default graph

g = tf.Graph()              # API!! Create a new graph
with g.as_default():        # API!!
  c1 = tf.constant(0.1)     #
  c2 = tf.constant(0.3)
  c3 = tf.add(c1, c2)
  print(c1.graph)   #Graph g
  print(g)          #Graph g
  print(c.graph)    #Default graph

g2 =  tf.get_default_graph()  # API!! Default graph
print(g2)

tf.reset_default_graph()
g3 =  tf.get_default_graph()    #New different graph
print(g3)

#Get tensor
print("\r\nGet Tensor")
print(c1.name)
t = g.get_tensor_by_name(name = "Const:0")      #API!!
print(t)

#Get operation
print("\r\nGet Operation")
a = tf.constant([[1.0, 2.0]])
b = tf.constant([[1.0], [3.0]])

tensor1 = tf.matmul(a, b, name='exampleop')
print(tensor1.name,tensor1)
test = g3.get_tensor_by_name("exampleop:0")
print(test)

print(tensor1.op.name)                      #print the operation name
testop = g3.get_operation_by_name("exampleop")      #Get the operation through the name
# print(testop)

with tf.Session() as sess:
    test =  sess.run(test)
    print(test) 
    test = tf.get_default_graph().get_tensor_by_name("exampleop:0")
    print (test)

tt2 = g.get_operations()    #Get all operation of this graph
print(tt2) #[<tf.Operation 'Const' type=Const>, <tf.Operation 'Const_1' type=Const>, <tf.Operation 'Add' type=Add>]