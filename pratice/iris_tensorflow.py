import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import DataFrame, Series
import seaborn as sns


#tf.contrib.learn 을 활용한 데이터 분석
path = 'D:\\github\\tensorflow\\pratice\\iris_data\\'
IRIS_TRAINING = 'iris_training.csv'
IRIS_TEST = "iris_test.csv"

analysis_data = pd.read_csv(path+'iris_training.csv')
analysis_data.index.name = 'id'
analysis_data.columns = ['SepalLengthCm','SepalWidthCm',
                         'PetalLengthCm','PetalWidthCm', 'Species']
analysis_data.Species[analysis_data.Species == 0] = 'a'
analysis_data.Species[analysis_data.Species == 1] = 'b'
analysis_data.Species[analysis_data.Species == 2] = 'c'


analysis_data.head()

g = sns.pairplot(analysis_data, hue="Species", size=2.5)
Callback

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=path+IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=path+IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)#특징 데이터는 float32 타입

#column_name: A string defining real valued column name.
#dimension:  An integer specifying dimension of the real valued column 4
#데이터 셋에 있는 특성의 데이터 타입을 지정.
#4개의 특성이므로 dimensions = 4
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

#DNN 모델을 생성
#은닉층 10, 20, 10
#분류는 3가지
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,#4dim
                                            hidden_units=[10, 20, 10],#3 layers
                                            n_classes=3,#3 classes
                                            model_dir=path+"iris_model")#
classifier.fit(x=training_set.data,#특성 데이터
               y=training_set.target,#타깃 값
               steps=100)

#test data and test label is input
accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)
accuracy_score['accuracy']

#임의로 데이터 생성
new_samples = np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)

y = list(classifier.predict(new_samples, as_iterable=True))
str(y)
classifier
