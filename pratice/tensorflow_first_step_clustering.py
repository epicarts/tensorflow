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
#random_shuffle 첫번째 텐서의 차원을 기준으로 (vector)의 원소를 무작위로 섞음.
#k개의 중심을 선택.
centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0,0], [k,-1]))

vectors.get_shape()#순서 대로 D0 D1 이렇게 부름. (2000 X 2)
centroids.get_shape()#만약 차원이 하나 더 늘어나면 D0 D1 D2 로 바뀜. (4 X 2)

#이제 vectors 와 centroids 사이의 거리를 구해야함. but 둘다 2D 이지만, 계산하기에 맞지가 않음.




#vector 한차원 더 확장 시킴 (1, 2000, 2)
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_vectors

#centroid 도 한차원 더 확장시킴 (4, 1, 2)
expanded_centroid = tf.expand_dims(centroids, 1)
expanded_centroid

subtract_result = tf.subtract(expanded_vectors, expanded_centroid)
subtract_result.get_shape()

#여기까지 유클리드 제곱을 사용하여 거리를 구함.
sqr = tf.square(subtract_result)
sqr.get_shape()

#특정 차원을 제거하고 합계를 구함. (x, y)좌표의 값 차원을 없애고 더함
#즉, (4 X 2000) 배열에 모두 (x + y)를 함
distances = tf.reduce_sum(sqr, 2)
distances.get_shape()

#argmin최소값의 인덱스를 얻기 위해서 사용.
#(4 X 2000) 에서 4가 최소가 되는 값만 남겨두고 다 버림. => 1차원으로 만듬.
assignments = tf.argmin(distances, 0)
assignments


'''
중심 mean 구하기
'''
#k(4)개의 개수만 큼 for 문을 돔. c = 1, 2, 3, 4 .... k
#tf.equal(assignments, c) 군집과 매칭되는 텐서의 각원소 위치를 True 로 표시한 bool 텐서
bool_tensor = tf.equal(assignments, 1)#2,3,4 ... k개
bool_tensor

#where 함수를 사용하여, 불리언에 Ture로 표시된 위치를 값으로 가지는 텐서(2000 X 1)를 만듬
where_Ture = tf.where(bool_tensor)
where_Ture

#reshape로 (2000 x 1) 을 (1 x 2000) 으로 변경
tf.reshape(where_Ture, [1, -1])

#gather 함수를 사용하여 (1 X 2000)을 k번째 군집에 속한 모든점(x, y)를 가진 텐서를 만듬
#(1 X 2000 X 2)
tf.gather()

#reduce_mean을 사용하여 k번째 군집에 속한 모든 점의 평균 값을 가진 텐서 (1 X 2)를 만듬. 2 = (x,y)
tf.reduce_mean()


#리스트를 만들어서 tf.concat([리스트]) 에 넣음. / 리스트를 이어 붙여서(concat) 텐서를 만듬.
#최종코드
means = tf.concat([tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])), reduction_indices=[1]) for c in range(k)], 0)

means#(총 군집개수 X x,y)  (4 X 2) 사이즈의 텐서가 나옴.

'''
이제 중심을 means 텐서의 새 값으로 업데이트 하는 코드를 작성해야함
'''
#assigning 할당하다. 중심값과 means 를 update_centroids에 할당함
update_centroids = tf.assign(centroids, means)
update_centroids

#variable값을 쓰기전에 항상 초기화
init_op = tf.global_variables_initializer()

sess = tf.Session()

for step in range(100):
    #매 반복마다 중심은 업데이트 되고 각 점은 새롭게 군집에 할당됨.
    #update_centroids(4, 2): centroids, means 을 할당한 텐서 / 연상ㄴ은 리턴이 없음
    #centroids(4, 2): 내가 임의로 정한 군집
    #assignments(2000,): assignments 군집들이 최소가 되는 값들만 모아둔 텐서
    _, centroid_value, assignment_values = sess.run([update_centroids, centroids, assignments])
    #update_centroids 연산은 리턴 값이 없음.

centroids
update_centroids
assignments
