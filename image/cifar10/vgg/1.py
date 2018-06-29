import numpy as np
from cifar10_input import C10Input

c10input = C10Input('/tmp/cifar-10-batches-py')

data, label = c10input.get_batch_data(10)

print(type(data))
print(np.shape(data))
# print(data)
print(np.shape(label))