import tensorflow as tf
import numpy as np
'''
군집화 (unsupervised learning)
k-mean clustering

tensor???
동적 크기를 갖는 다차원 데이터 배열. (bool, string, int16, float32....)
'''
#랭크(rank): 텐서에서 배열의 차원 / rank = 0 scalar / rank = 1 vector
#(3 X 3) 행렬 / rank = 2
t = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
'''
#shape, rank dimension, number
[], 0,  0-D
[D0], 1, 1-D
[D0, D1], 2, 2-D
[D0, D1, D3], 3, 3-D

shape(텐서의 구조), size(텐서의 크기), rank(텐서의 랭크), reshape(구조를 바꿈)
squeeze(크기가 1인 차원삭제), expand_dims(차원 추가), slice(텐서 일부분 삭제)
split(차원을 기준으로 여러텐서로 나눔) tile(여러번 중복하여 새 텐서 생성)
concat(한차원을 기준으로 텐서를 이어 붙임) reverse(차원을 역전시킴)
transpose(텐서를 전치) gather(주어진 인덱스에 따라 텐서의 원소를 모음)
'''
#(2000 X 2) rank = 2 텐서를 3차원으로 변경.
points = np.zeros((2000, 2))
vectors = tf.constant(points)
#한차원 확대 / 인수 부분에 0을 넣음./ 크기는 지정 못함
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_vectors#제일 앞부분에 차원 추가 (0, 2000, 2)

expanded_vectors2 = tf.expand_dims(vectors, 1)
expanded_vectors2#중간 부분에 차원 추가 (0, 2000, 2)

'''
텐서플로우에서 데이터를 얻는 방법 3가지
1. 데이터 파일로부터 얻기
2. 상수나 변수로 미리 로드하기
3. 파이선 코드로 작성해 제공하기

Iterator.get_next() tf.data.Dataset
'''
#크기가 크지 않은 경우 메모리에 데이터를 미리 로드 할 수 있음.

#상수 생성
tf.constant#함수 인수로 지정된 값을 이용하여 상수 텐서를 생성
tf.zeros_like #모든 원소를 0으로 초기화한 텐서 생성
tf.ones_like # 모든 원소를 1로 초기화한 텐서 생성
tf.fill # 주어진 스칼라 값으로 원소를 초기화한 텐서를 생성

#텐서 생성
tf.random_normal # 정규분포를 따르는 난수로 텐서를 생성
tf.truncated_normal #정규분포를 따르는 난수로 텐서를 생성, 표준편차의 2배수보다 큰값 제거
tf.random_uniform #균등 분포를 따르는 난수로 텐서를 생성
tf.random_shuffle #첫번쨰 차원을 기준으로 텐서의 원소를 섞음
tf.set_random_seed #난수 시드(seed)를 설정
#텐서의 차원 구조를 매개 변수로 받아야함.

#변수 생성.
tf.variable()
#변수를 사용하려면 데이터의 그래프를 구성한 후 run() 함수를 실행하기 전 반드시 초기화.
tf.global_variables_initializer()#를 사용해서 초기화 했음.
#변수 모델을 훈련 시키는 도중 or 훈련후 tf.train.Saver 클래스를 이용하여 저장가능

#symbolic 변수 / 프로그램 실행 중에 데이터를 변경하기 위해서 사용
tf.placeholder() # 원소의 자료형, 텐서의 구조, 이름을 매개변수로 줄 수 있음.

#다음 메서드를 호출할때 feed_dict 매개변수로 플레이스 홀더를 지정하여 전달할 수 있음
Tensor.eval()
Session.run()

sess.run(y, feed_dict={a: 3, b: 3})

'''
k-mean 평균 알고리즘
centroid 라고 부르는 k개의 dot 으로 나눠짐.
군집을 구성할 때 직접 오차함수를 최소화하려면 계산 비용이 매우 많이 듬.(NP-난해)
NP-난해? 입력 값이 많아질수록 알고리즘을 수행하는데 걸리는 시간이 늘어남.
휴리스틱 기법을 사용할 거임.

반복 개선 기법(iterative refinement)
1. 초기단계: K개의 중심의 초기 집합을 결정
2. 할당단계: 각데이터를 가장 가까운 군집에 할당
3. 업데이트 단계: 각 그룹에 대해 새로운 중심을 계산

1. 임의로 설정하겠음.
2. 알고리즘이 수렴되었다고 간주될때까지 루프를 통해 반복.
3. 이 알고리즘은 휴리스틱 방법임으로 진짜 최적값으로 수렴한다는 보장은 없음.
4. 결과는 초기 중심값을 어떻게 정했느냐에 따라 영향을 많이 받음.
5. 이 알고리즘 속도는 매우 빠르므로 여러 번 알고리즘을 수행하여 결과들을 비교
'''
import numpy as np


num_points = 2000
vectors_set = []

for i in range(num_points):
    if np.random.random() > 0.5:#랜덤으로 만든게 0.5 이상이면,
        vectors_set.append([np.random.normal(0.0, 0.9),
                            np.random.normal(0.0, 0.9)])
    else:
        vectors_set.append([np.random.normal(3.0, 0.5),
                            np.random.normal(1.0, 0.5)])


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df = pd.DataFrame({"x": [v[0] for v in vectors_set],
                   "y": [v[1] for v in vectors_set]})
df
sns.lmplot("x", "y", data=df, fit_reg=False, height=6)

'''
본격적인 군집화 알고리즘 시작
'''
#vectors_set(2000 X 2) 를 상수화 시켜서 vector에 담음 / 모든 데이터를 텐서로 옮김
vectors = tf.constant(vectors_set)
k = 4

#중심은 변수(Variable) 생성.
#random_shuffle 첫번째 텐서의 차원을 기준으로 (vector)의 원소를 섞음.
centroid = tf.Variable(tf.random_shuffle(vectors), [0,0], [k,-1])
centroid
vectors.get_shape()#순서 대로 D0 D1 이렇게 부름.
centroid.get_shape()#만약 차원이 하나 더 늘어나면 D0 D1 D2 로 바뀜.

#vector 한차원 더 확장 시킴 (1, 2000, 2)
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_vectors

#centroid 도 한차원 더 확장시킴 (2000, 1, 2)
expanded_centroid = tf.expand_dims(centroid, 1)
expanded_centroid


#argmin: Returns the index with the smallest value across dimensions of a tensor.
#주어진 expanded_vectors(1, 2000, 2) - expanded_centroid(2000, 1, 2) 뺄셈을 하고
#제곱 한거를 모두 더하는데(reduce_sum) 차원의 크기를 2로 나누어서 더함
assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(
    expanded_vectors, expanded_centroid)), 2), 0)
