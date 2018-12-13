from keras.utils.data_utils import get_file
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import os
import requests
import base64

# Encode text values to indexes(i.e. [1],[2],[3] for red,green,blue).
def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_

# Encode a numeric column as zscores
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd

# Encode text values to a single dummy variable.  The new columns (which do not replace the old) will have a 1
# at every location where the original column (name) matches each of the target_values.  One column is added for
# each target value.
def encode_text_single_dummy(df, name, target_values):
    for tv in target_values:
        l = list(df[name].astype(str))
        l = [1 if str(x) == str(tv) else 0 for x in l]
        name2 = "{}-{}".format(name, tv)
        df[name2] = l

# Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

# Encode a column to a range between normalized_low and normalized_high.
def encode_numeric_range(df, name, normalized_low=-1, normalized_high=1,data_low=None, data_high=None):
    if data_low is None:
        data_low = min(df[name])
        data_high = max(df[name])
    df[name] = ((df[name] - data_low) / (data_high - data_low)) \
               * (normalized_high - normalized_low) + normalized_low

# Encode a column to a range between normalized_low and normalized_high.
def encode_numeric_range(df, name, normalized_low=-1, normalized_high=1,data_low=None, data_high=None):
    if data_low is None:
        data_low = min(df[name])
        data_high = max(df[name])
    df[name] = ((df[name] - data_low) / (data_high - data_low)) \
               * (normalized_high - normalized_low) + normalized_low

path = get_file('kddcup.data_10_percent.gz',
                origin='http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz')
print(path)

df = pd.read_csv(path, header=None)
print("Read {} rows.".format(len(df)))
df.head()
df.columns = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'outcome'
]
df[0:5]
import tensorflow.contrib.learn as skflow
import pandas as pd
import os
import numpy as np
from sklearn import metrics
from scipy.stats import zscore
path = "./data/"
filename_read = os.path.join(path,"auto-mpg.csv")
filename_read
filename_read
# Now encode the feature vector
ENCODING = 'utf-8'
df
encode_numeric_zscore(df, 'duration')
encode_text_dummy(df, 'protocol_type')
encode_text_dummy(df, 'service')
encode_text_dummy(df, 'flag')
encode_numeric_zscore(df, 'src_bytes')
encode_numeric_zscore(df, 'dst_bytes')
encode_text_dummy(df, 'land')
encode_numeric_zscore(df, 'wrong_fragment')
encode_numeric_zscore(df, 'urgent')
encode_numeric_zscore(df, 'hot')
encode_numeric_zscore(df, 'num_failed_logins')
encode_text_dummy(df, 'logged_in')
encode_numeric_zscore(df, 'num_compromised')
encode_numeric_zscore(df, 'root_shell')
encode_numeric_zscore(df, 'su_attempted')
encode_numeric_zscore(df, 'num_root')
encode_numeric_zscore(df, 'num_file_creations')
encode_numeric_zscore(df, 'num_shells')
encode_numeric_zscore(df, 'num_access_files')
encode_numeric_zscore(df, 'num_outbound_cmds')
encode_text_dummy(df, 'is_host_login')
encode_text_dummy(df, 'is_guest_login')
encode_numeric_zscore(df, 'count')
encode_numeric_zscore(df, 'srv_count')
encode_numeric_zscore(df, 'serror_rate')
encode_numeric_zscore(df, 'srv_serror_rate')
encode_numeric_zscore(df, 'rerror_rate')
encode_numeric_zscore(df, 'srv_rerror_rate')
encode_numeric_zscore(df, 'same_srv_rate')
encode_numeric_zscore(df, 'diff_srv_rate')
encode_numeric_zscore(df, 'srv_diff_host_rate')
encode_numeric_zscore(df, 'dst_host_count')
encode_numeric_zscore(df, 'dst_host_srv_count')
encode_numeric_zscore(df, 'dst_host_same_srv_rate')
encode_numeric_zscore(df, 'dst_host_diff_srv_rate')
encode_numeric_zscore(df, 'dst_host_same_src_port_rate')
encode_numeric_zscore(df, 'dst_host_srv_diff_host_rate')
encode_numeric_zscore(df, 'dst_host_serror_rate')
encode_numeric_zscore(df, 'dst_host_srv_serror_rate')
encode_numeric_zscore(df, 'dst_host_rerror_rate')
encode_numeric_zscore(df, 'dst_host_srv_rerror_rate')
outcomes = encode_text_index(df, 'outcome')
num_classes = len(outcomes)

df[0:5]
df.dropna(inplace=True,axis=1) # 결측값 제거
df[0:5]

# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)
    else:
        # Regression
        return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.float32)

x, y = to_xy(df,'outcome')


from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping

# 0.25 퍼센트를 테스트 데이터로 둠. 랜덤 시드 값은 42로 고정
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
model = Sequential()# 모델을 생성
x.shape
y.shape[1]

#왜 코드 이따구로 짯지 ? 인풋값과 아웃풋 값이 다른데 ?

#Dense 레이어는 입력 뉴런 수에 상관없이 출력 뉴런 수를 자유롭게 설정
#activation 으로는 softmax, sigmoid, tanh, relu 가 있음.
#첫번째 층으로 활성화 함수relu rectifier  사용. It is mainly used of the activation funition of the hidden layer.
#가중치 초기화 방법 : normal 가우시안 분포 = 정규분포 / 총 10개의 노드.
#입력 뉴런 개수: 120 개 input_dim=x.shape[1], 출력 뉴런 수 10
model.add(Dense(10, input_dim=x.shape[1], kernel_initializer='normal', activation='relu'))

#두번째 층 Regular densely-connected neual network layer.
#가중치 초기화 방법 : normal 가우시안 분포 = 정규분포 / 총 50개의 노드
model.add(Dense(50, input_dim=x.shape[1], kernel_initializer='normal', activation='relu'))

#가중치 초기화 방법 : normal 가우시안 분포 = 정규분포 / 총 10개의 노드
model.add(Dense(10, input_dim=x.shape[1], kernel_initializer='normal', activation='relu'))

#가중치 초기화 방법 : normal 가우시안 분포 = 정규분포
model.add(Dense(1, kernel_initializer='normal'))

#마지막 층은 클래스별로 확률 값이 나오도록 출력시킵니다. softmax 이 확률값을 모두 더하면 1이 됩니다.
#input 은 비어있으니까 전에 만든 모델의 개수를 따라가고, y값은 23개 출력 총 23개의 아웃풋이 나옴
model.add(Dense(y.shape[1],activation='softmax'))

#모델 학습과정 설정
#optimizer=adam : 최적의 가중치를 검색하는 데 사용되는 최적화 알고리즘으로 효율적인 경사 하강법 알고리즘
#loss :현재 가중치 세트를 평가하는 데 사용한 손실 함수 다중 클래스 문제이므로 categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer='adam')

#조기 종료  더 이상 개선의 여지가 없을 때 학습을 종료
#min_delta : 개선되고 있다고 판단하기 위한 최소 변화량
#patience : 개선이 없다고 바로 종료하지 않고 개선이 없는 에포크를 얼마나 기다려 줄 것인 가를 지정
#auto : 관찰하는 이름에 따라 자동으로 지정합니다.
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

#모델을 학습 시킨다.  verbose디버깅 epochs = 반복 횟수.
#validation_data 유효성 확인 테스트 데이터
model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor],verbose=2,epochs=5)


np.unique(y_test)# y 데이터는 0 or 1 밖에 없음. 머지 시발?
model.get_layer

# Measure accuracy
pred = model.predict(x_test)
pred = np.argmax(pred,axis=1)
y_eval = np.argmax(y_test,axis=1)
score = metrics.accuracy_score(y_eval, pred) #점수 내는거.
print("Validation score: {}".format(score))
