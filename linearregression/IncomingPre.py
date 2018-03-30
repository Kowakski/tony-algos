import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()

TrainDataPath = "../woodata/adult.data"
TestDataPath = "../woodata/adult.test"

mark = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "gender", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
FIELD_DEFAULTS = [ [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], [""], [""]]

def __parse_line( line ):
    feature = tf.decode_csv( line, record_defaults =  FIELD_DEFAULTS )
    # feature = tf.Print(feature, [feature])
    feature = dict(zip(mark, feature))
    label = feature.pop( "income" )
    print("FEATURE IS**:{0}\r\n".format(feature))
    return feature, label

#define input function
def data_input_fun( path, batchsize ):
    if not os.path.exists(path):
        return
    data = tf.data.TextLineDataset(path)
    data.map(__parse_line)
    return data

#define feature columns
#convert string to interger
age = tf.feature_column.numeric_column(
    key = 'age' )

education_num = tf.feature_column.numeric_column(
    key = 'education_num' )

capital_gain = tf.feature_column.numeric_column(
    key = 'capital_gain' )

capital_loss = tf.feature_column.numeric_column(
    key = 'capital_gain' )

hours_per_week = tf.feature_column.numeric_column(
    key = 'hours_per_week' )

workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    key = 'workclass',
    vocabulary_list = [ 'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
        'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked' ] )

fnlwgt = tf.feature_column.categorical_column_with_vocabulary_list(
    key = '',
    vocabulary_list = [] )
education = tf.feature_column.categorical_column_with_vocabulary_list(
    key = 'education',
    vocabulary_list = ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
        'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
        '5th-6th', '10th', '1st-4th', 'Preschool', '12th'] )

marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
    key = 'marital_status',
    vocabulary_list = ['Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
        'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'] )

occupation = tf.feature_column.categorical_column_with_vocabulary_list(
    key = '',
    vocabulary_list = ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', '?', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv'] )

relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    key = '',
    vocabulary_list = ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
        'Other-relative'] )

# race = tf.feature_column.categorical_column_with_vocabulary_list(
#     key = '',
#     vocabulary_list = [] )

gender = tf.feature_column.categorical_column_with_vocabulary_list(
    key = 'gender',
    vocabulary_list = ['Male','Female'] )

native_country = tf.feature_column.categorical_column_with_vocabulary_list(
    key = 'native_country',
    vocabulary_list = [] )

income = tf.feature_column.categorical_column_with_vocabulary_list(
    key = '',
    vocabulary_list = [] )

#define estimator

#evaluate
data = data_input_fun(TrainDataPath, 500)
iterator = tfe.Iterator(data)
# for  one_item in iterator:
#     print( one_item )
