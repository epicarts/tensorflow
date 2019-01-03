import numpy as np

num_points = 200
vectors_set = []

for i in range(num_points):
    x = np.random.normal(5,5)+15#
    y = x*1000 + (np.random.normal(0,3))*1000
    vectors_set.append([x,y])
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]


import matplotlib.pyplot as plt
plt.plot(x_data, y_data,'ro')

import tensorflow as tf
'''
1. 랜덤으로 W 값 초기화
2.
'''
#텐서의 차원 1, -1.0 ~ 1.0 의 값으로 랜덤하게 초기화
random_val = tf.random_uniform([1],-1.0,0.0)
W = tf.Variable(random_val)#W의 초기값은 -1 ~ 1사이의 1차원 값으로 랜덤하게 초기화 한다.
b = tf.Variable(tf.zeros([1]))#텐서의 1차원 0으로 정의
y_pred = W * x_data + b# 예측 y값 = W * x_data + b
#x를 넣었을때 예측값 - 원래값을 제곱. => 전체 값의 평균을 구함.
loss = tf.reduce_mean(tf.square(y_pred - y_data)) # (예측값 - 원래값) 제곱
optimizer = tf.train.GradientDescentOptimizer(0.0015)# 경사하강법: 학습 속도 learning rate\
#위의 경사 하강법(학습룰 0.0015)을 이용하여 loss(코스트함수)가 최소화 되는 값을 찾아야함.
train = optimizer.minimize(loss)
init = tf.initialize_all_variables()#모든 값 초기화

sess = tf.Session() #텐서플로우 세션 만들기
sess.run(init)#초기값 init 로 세션 시작


for step in range(1000):
    sess.run(train)
    print(step, sess.run(W),sess.run(b))
    print(step, sess.run(loss))

    plt.plot(x_data,y_data,'ro')
    plt.plot(x_data, sess.run(W)*x_data + sess.run(b))



wirter = tf.summary.FileWriter('testgraph',sess.graph)

#코스트 함수의 개념  = 예측값과 실제 값 사이의 차이를 계산해주는 함수.
# 예측된 값과 입력된 값들의 차이에 대한 평균 값을 구함.

#선형 함수에서 코스트
cost = sum((y_pred - y_origin)^2))/n #예측값 - 원래값 제곱을 전부 더하고 n으로 나눔
#sum/n 을 하는 이유는 전체 값의 평균을 구하기 위함.

#sum 값을 전부 더하고 n으로 나눔 = > 전체 값의 평균을 구하기 위함
#   -(원래값) * log(예측값)  - (1-원래값) * log(예측값)
costfunction=
    (1/n) * sum(
    -y_origin * log(sigmoid(Wx + b)) -(1 - y_origin) * log(1-(sigmoid(Wx + b))))

exp(​ ​-(W*x+b)​ ​)​ ​(즉​ ​e^(-Wx+b)​ ​)
#시그모이드 함수에 저 e가 씌여지기 때문에  e를 상쇄할 수 있는 역치 함수로 log를 씌움
#e가 들어간 코스트 함수self.

#sigmoid 를 풀어 쓰면 이렇게 나옴.
sigmod(Wx + b) = 1 / (1 + exp(-(Wx + b)))

import tensorflow as tf
a = tf.constant([5],dtype = tf.float32)
a
b = tf.constant([10],dtype = tf.float32)
b
c = tf.constant([2],dtype = tf.float32)
c
d = a*b+c
d
print(d)

sess2 = tf.Session()#세션 생성.
result = sess2.run(d)#그래프 d를 실행하도록 run 함
result
wirter = tf.summary.FileWriter('testgraph',sess2.graph)

tf.placeholder(dtype,shape,name)# 학습용 데이터를 담는 그릇.플레이스 홀더
#dtype 데이터형 . shape 행렬의 차원[3,3] , name 이것의 이름.
#input :x  = placeholder
x = tf.placeholder(dtype = tf.float32)

# y = x *2 그래프
input_data = [1,2,3,4,5]
y = x * 2

sess3 = tf.Session()
result = sess3.run(y ,feed_dict={x:input_data})#x:[1,2,3,4,5] 하나씩 읽음

#x 가 입력 데이터라고 했을 때, W b 는 변하는 가변 값.
tf.Variable.__init__(initial_value=None,trainable=True,collections=None,
                     validate_shape=True,caching_device=None,name=None,
                     variable_def=None,dtype=None,expected_shape=None,
                     import_scope=None)

var = tf.Variable([1,2,3,4,5],dtype=tf.float32)
var

x = tf.placeholder(dtype=tf.float32)#데이터를 저장할 X 공간 생성
W = tf.Variable([2],dtype=tf.float32)# 가변으로 변하는 값들을 저장할 공간 생성/ 2
W
y = W*x

sess4 = tf.Session()# 세션 만들기
result = sess4.run(y,feed_dict={x:input_data})# 계산 할 식 / 딕셔너리 형태로 {x:[1,2,3,4,5]}
#result = sess4.run(W*x,feed_dict={x:input_data})
#작동하기 전에 세션을 초기화 해야함. 이대로는 작동이 불가능함

result

init = tf.global_variables_initializer()
sess4.run(init)
result = sess4.run(y,feed_dict={x:input_data})# 계산 할 식 / 딕셔너리 형태로 {x:[1,2,3,4,5]}

result



x = tf.constant([[1.0, 2.0, 3.0]]) # 2차원 1행
x.get_shape()
# (1,3) 1행 3열 2차원
w = tf.constant([[2.],[2.],[2.]],dtype=tf.float32)
w.get_shape()

y = tf.matmul(x,w)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
result = sess.run(y)
result

x2 = tf.placeholder(dtype=tf.float32, shape=[None,3])# n,3 사이즈 그릇을 만듬
x2
x
w2 = tf.Variable([[2.],[2.],[2.]], dtype=tf.float32)

y2 = tf.matmul(x2,w2)

sess5 = tf.Session()
init = tf.global_variables_initializer()
sess5.run(init)

input_data = [[1,2,3]]
result = sess5.run(y2,feed_dict={x2:input_data})# y2 공식에 x2그릇에 inpht_data를 넣어 피드함.

x2
result


input_data = [[1,1,1],
              [2,2,2]] # 2row 3coulm

x = tf.placeholder(dtype = tf.float32, shape = [2,3])

w = tf.Variable([[2],[2],[2]], dtype=tf.float32)
b = tf.Variable([4],dtype = tf.float32)
x.get_shape()
w.get_shape()
b.get_shape()

y = tf.matmul(x,w) + b
x.get_shape()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
result = sess.run(y, feed_dict={x:input_data})

result.shape
print(result)


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data',one_hot=True)


# 55000 x 784  *  784 X 10 +
# 55000개의 파일. 784개의 인풋. 10개의 아웃풋.
#softmax(Wx+b)  _y = 실제값
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.matmul(x,W)+b,y_))


x = tf.placeholder(tf.float32,[None, 784])
W = tf.Variable(tf.zeros([784,10])) #(784,10) 크기로 초기화
b = tf.Variable(tf.zeros([10]))# (,10) 크리
b.get_shape()
W.get_shape()

#x값으로 학습을 시켜서 0~9를 가장 잘 구별해내는 W,b 값을 찾아야함.
#W*b 를하면 55000 X 10 사이즈가 됨. b는 10이지만 자동으로 55000 X 10 사이즈로 변환
#softmax 는 모든 결과값이 0 ~ 10 사이의 값이 나옴

#실제코드
x = tf.placeholder(tf.float32,[None,784]) #(?,784) 사이즈 그릇 생성
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
k = tf.matmul(x,W) + b #선형 공식

#sum(0~10 사이의 값) = 1
y = tf.nn.softmax(k) # 위 가설식을 softmax 에 넣기.


#감소 정의 및 조정.
y_ = tf.placeholder(tf.float32, [None,10]) # (?, 10) 사이즈 그릇 생성
learning_rate = 0.5 # 학습률 0.5

# ax + b를 이용한 예측값 y 와 라벨링 되어있는 y를 이용.
#softmax(ax + b) 해야되지만 자동으로 해주기 떄문에 생략.
#logits 10 , labels = 10
#평균값 reduce_mean 구하기.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= k,
                                                               labels = y_))

#경사하강법을 이용함. 코스트가 최소화 되는 값을 찾음.
# 코스트= softmax entropy 를 이용해 실제 값과 예측값 을 비교하여 나온 값.
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for _ in range(1000):
    #1000번씩 전체데이터에서 100개씩 뽑아서 트레이닝을 함.
    batch_xs , batch_ys = mniszt.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

sess.run(b)
sess.run(W)

W.get_shape()

#테스트 하는코드
#행렬 y에서 몇번째에 가장 큰 값이 들어가 있는지 리턴 해 주는 함수
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
session = tf.InteractiveSession()
data = tf.constant([9,2,11,4])
idx = tf.argmax(data,0)
idx.eval()

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(accuracy, feed_dict = {x:mnist.test.images, y_:mnist.test.images})
tf.argmax(y_,1)
tf.argmax(y,1)

import tensorflow as tf

x = tf.placeholder(tf.float32,[None, 784]) # 50000 x 784 크기 배열
W = tf.Variable(tf.zeros([784,10]))
b = tf.Vari

mnist.test.images[0]
mnist.test.labels

import numpy as np
data = np.array([[56.0, 0.0, 4.4, 68.0],
          [1.2 , 104.0, 52.0, 8.0],
          [1.8, 135, 99.0, 0.9]])

x = tf.placeholder(tf.float,[])
cal = data.sum(axis = 0)
cal
percentage = 100 * data / cal.reshape(1,4)
percentage
percentage = 100 * data / cal #reshape 쓰는걸 권장
percentage

a = np.random.randn(5)
a
a.shape
print(a.T)
np.dot(a,a.T)

a = np.random.randn(5,1)
print(a)
print(a.T)
a.shape
a.T.shape
np.dot(a,a.T)#5X1 * 1X5
np.dot(a.T,a)


a = []
x = np.linspace(-8,8,100)
y = [(1/(1+np.exp(-i))) for i in x]
y
import numpy as np
import matplotlib.pylab as plt
%matplotlib inline
plt.plot(x,y)

y2 = [((1/(1+np.exp(-i)))*(1 - (1/(1+np.exp(-i))))) for i in x]
y2
plt.plot(x,y2)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(x,y)
ax.plot(x,y2)

plt.show()
import matplotlib.pyplot as plt
plt.plot(range(100))
plt.show()
import math
y = [math.tan(i) for i in x]


x = np.linspace(0,8,100)
y = [np.sin(i) for i in x]
plt.plot(x,y)

x = np.linspace(0,8,100)
y = [np.cos(i) for i in x]
plt.plot(x,y)

x = np.linspace(-8,8,100)
y = [(abs(i) * i > 0) for i in x]
plt.plot(x,y)

x = np.linspace(-8,8,100)
y = [max(0,i) for i in x]
plt.plot(x,y)
x = np.linspace(-8,8,100)
y = [max(0.01*i,i) for i in x]
plt.plot(x,y)



a= np.random.randn(3,2)* 1001#너무 큰값을 가지면 안댐. 활성화 함수에서 경사의 기울기가 매우 낮음
a
b = np.zeros((3,1))
b


#입력 특징 3
# 은닉층 5 5 3
#출력 1

tf
