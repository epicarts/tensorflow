import tensorflow as tf

#프로그램 실행중에 값을 변경할 수 있는 'symbolic' 변수 정의
a = tf.placeholder("float")
b = tf.placeholder("float")


#multiply 에서 매개 변수(placeholder 형태)를 전달
#텐서를 조작 하기 위해 텐서플로가 제공하는 수학 연산 함수
y = tf.multiply(a, b)

'''
덧셈(add) 뺄셈(subtract) 곱셈(multiply) 나눗셈몫(div) 나머지(mod) 절대값(abs)
negative(음수) 부호(sign) 역수(reciprocal) 제곱(squre) 반올림(round) 제곱근(sqrt)
거듭제곱(pow) 지수(exp) 로그(log) maximum(최대값) 최소값(minimum) 코사인(cos)
사인(sin)
https://goo.gl/sUQy7M


대각행렬(diag)  전치행렬(transpose) 행렬곱(matmul) 정방행렬의 행렬식(matrix_determinant)
정방행렬의 역행렬(matrix_inverse)
'''
#세션을 생성. 텐서플로 라이브러리와 프로그램간의 상호작용.
sess = tf.Session()

#세션 생성 => run() 메서드를 호출 할때 .symbolic 코드가 실제로 실행.
#run 메서드에 feed_dict 인수로 변수의 값을 넘겨줌.
sess.run(y, feed_dict={a : 3, b: 3})

#즉, 전체 알고리즘 기술 => 세션을 생성하여 연산을 실행.
#IPython 같은 대화 환경 => tf.InteractiveSession 클래스도 제공함.

#그래프 구조 : 수학 계산을 표현함. 연산에 대한 모든정보를 담고 있음.
#노드: 수학 연산을 나타냄. 데이터의 입력과 출력의 위치를 나타냄. 저장된 변수를 읽거나 씀.
#에지: 노드사이의 관계를 표현, 텐서를 운반함.

#그래프 구조로 표현된 정보를 이용하여, 트랙잭션간의 의존성을 인식 => 디바이스에 비동기적으로 병렬적 연산을함.
#병렬처리 => 빠르게 알고리즘 실행. 복잡한 연산을 효율적으로 처리.
'''
주요 연산 커널
Add, Sub, Mul, Div, Exp, Log, Greater, Less, Equal
concat, Slice, split, constant, Rank, Sahpe, shuffle
MatMul, MatrixInverse, matrix_determinant
SoftMax, Sigmoid, ReLU, Convoution2D, MaxPool
Save, Restore
Enqueue, Dequeue, MuteAcquire, MutexRelease
Merge, Switch, Enter, Leave, NexIteration
'''

#텐서보드(TensorBoard): 프로그램을 최적화, 디버깅 기능, 그래프의 상세정보와 매개변수들에 대한 여러 통계 데이터.
#요약명령(summary operation): 텐서플로의 데이터, 추적 파일에 저장.

'''
선형 회귀 분석(linear regression)
독립변수 x, 상수항 b, 종속변수 y 사이의 관곌르 모델링 하는 방법.
두 변수 사이의 관계일 경우 단수 회귀라고도 함. 다중 회귀도 있음.
y = W*x + b
'''
import numpy as np


num_points = 1000
vectors_set = []


for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)#0을 기준으로 표준편차 0.55
    y1 = x1 * 0.1 + 0.3 +np.random.normal(0.0, 0.03
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]


import matplotlib.pyplot as plt


plt.plot(x_data, y_data, 'ro')

#Variable 메서드: 텐서플로 내부의 그래프 자료구조에 만들어질 하나의 변수를 정의
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

#우리가 생각하는 식
y = W * x_data + b

#비용함수 (coss function), 오차함수(error function)
#known = y_data / unknown = y
#시그마 (둘의 차이를 제곱)
loss = tf.reduce_mean(tf.square(y - y_data))

#train 안에 있는 여러 학습법 중에서 경사하강법 옵티마이져 클래스를 불러옴.
#경사 하강법을 쓰는 이유는 y - y_data 가 제곱이 되기 떄문.
optimizer = tf.train.GradientDescentOptimizer(0.5)#학습 속도: 0.5

#최적화(optimize)로는 loss가 최소가 되는 값을 찾아야함
#즉, 둘 사이의 오차가 최소가 되는 값을 찾음(경사 하강법 클래스 안에 있는 minimize)
train = optimizer.minimize(loss)

#모든 변수 초기화를 어떻게 할것인가?
init = tf.global_variables_initializer()

#세션 생성후 sess을 가동.=> 변수 초기값으로 init 를 파라미터로 넘김
sess = tf.Session()
sess.run(init)

#8번 반복함.
for step in range(8):
    #위에서 정의한 train에 따라서 세션을 run 시킴 / optimizer객체가 들어가면 train
    sess.run(train)
    #세션 안의 loss 를 보여줌 / 변수객체가 들어가면 값을 리턴
    print(step, sess.run(loss))

#세션 내부에서 작동한 W 와 b를 보여줌
sess.run(W)
sess.run(b)

plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
plt.show()
