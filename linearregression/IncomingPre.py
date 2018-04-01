import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# tfe.enable_eager_execution()

TrainDataPath = "/home/sln/share/now1/woodata/census_data/adult.data"
TestDataPath = "/home/sln/share/now1/woodata/census_data/adult.test"

mark = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "gender", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
FIELD_DEFAULTS = [ [0], [""], [0], [""], [0], [""], [""], [""], [""], [""], [0], [0], [0], [""], [""]]

def __parse_line( line ):
    feature = tf.decode_csv( line, record_defaults =  FIELD_DEFAULTS )
    # feature = tf.Print(feature, [feature])
    feature = dict(zip(mark, feature))
    label = feature.pop( "income" )
    print("FEATURE IS**:{0}\r\n".format(feature))
    return feature, tf.equal(label, '>50K')

#define input function
def data_input_fun( path, batchsize ):
    if not os.path.exists(path):
        return
    data = tf.data.TextLineDataset(path)
    data = data.map(__parse_line)
    data = data.batch(batchsize)
    return data

def create_feature_columns( ):
    #bucket columns
    SourceColumn = tf.feature_column.numeric_column( key = "age" )
    #0
    age = tf.feature_column.bucketized_column(
        source_column = SourceColumn, boundaries = [18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65] )

    #numerical columns
    #2
    fnlwgt = tf.feature_column.numeric_column(
        key = 'fnlwgt' )
    #
    education_num = tf.feature_column.numeric_column(
        key = 'education_num' )
    #
    capital_gain = tf.feature_column.numeric_column(
        key = 'capital_gain' )
    #
    capital_loss = tf.feature_column.numeric_column(
        key = 'capital_loss' )
    #
    hours_per_week = tf.feature_column.numeric_column(
        key = 'hours_per_week' )
    #1
    #vocabulary map
    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        key = 'workclass',
        vocabulary_list = [ 'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked' ] )

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
        key = 'occupation',
        vocabulary_list = ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', '?', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv'] )

    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        key = 'relationship',
        vocabulary_list = ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
            'Other-relative'] )
    #8
    race = tf.feature_column.categorical_column_with_vocabulary_list(
        key = 'race',
        vocabulary_list = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'] )
    #9
    gender = tf.feature_column.categorical_column_with_vocabulary_list(
        key = 'gender',
        vocabulary_list = ['Male','Female'] )

    #13
    native_country = tf.feature_column.categorical_column_with_vocabulary_list(
        key = 'native_country',
        vocabulary_list = ['United-States', 'Cuba', 'Jamaica', 'India', '?', 'Mexico', 'South', 'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran', 'Philippines', 'Italy', 'Poland', 'Columbia', 'Cambodia', 'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal', 'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala', 'China', 'Japan', 'Yugoslavia', 'Peru', 'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago', 'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary', 'Holand-Netherlands'] )

    # income = tf.feature_column.categorical_column_with_vocabulary_list(
    #     key = 'income',
    #     vocabulary_list = [] )
    ResultFeatureColumn = [ age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, gender, capital_gain, capital_loss, hours_per_week, native_country ]
    return ResultFeatureColumn
    #define estimator

def main(unused_argv):
    #create estimator
    estimator = tf.estimator.LinearClassifier( feature_columns = create_feature_columns(), n_classes = 2 )
    Train = estimator.train( input_fn = lambda:data_input_fun( TrainDataPath, 10 ), steps = 5 )
    evaluation = estimator.evaluate( input_fn = lambda:data_input_fun( TestDataPath, 10 ), steps = 1 )

if __name__ == "__main__":
    #evaluate
    # data = data_input_fun(TrainDataPath, 1)
    # iterator = tfe.Iterator(data)
    # for  one_item in iterator:
    #     print( one_item )
    tf.app.run()
