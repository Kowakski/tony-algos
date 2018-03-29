import os
import tensorflow as tf


TrainDataPath = "../woodata/adult.data"
TestDataPath = "../woodata/adult.test"

mark = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "gender", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
#define input function
def data_input_fun( path, batchsize ):
    if not os.path.exists(path):
        return
    data = tf.data.TextLineDataset(path)
    return data
    # print(data.output_classes)
    # print(data.output_shapes)
    # print(data.output_types)

#define feature columns

#define estimator

#evaluate

with tf.Session() as sess:
    data = data_input_fun(TrainDataPath, 500)
    iterator = data.make_one_shot_iterator()
    one_element = iterator.get_next()
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")
