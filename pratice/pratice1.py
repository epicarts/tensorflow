import tensorflow as tf

ta = tf.zeros((2,2))
#zeros(shape ,dtype=tf.float32, name=None)
#0 으로 초기화된 2,2 행렬 생성

print(ta.eval)
#세선을 설정하지 않은채 출력을 해서 오류가 뜸

session2 = tf.InteractiveSession()

print(ta.eval())
session2.close()

# 텐서값 출력은 eval 함수
# 함수 호출은 run 사용


W1 = tf.zeros((3,3)) #상수 텐서

W2 = tf.Variable(tf.zeros((2,2)), name = 'weights') # 변수 텐서 모든 요소값이 2*2 텐서

session = tf.InteractiveSession()
print(W1.eval())
print(W2.eval()) #초기화 하지 않았다는 에러가 뜸
#변수 초기화를 위해 global_variables_initializer 함수를 사용

session.run(tf.global_variables_initializer())# 초기화 함수를 run()함수의 인자로 넣어서 실행

session.close()

print(W1.eval()) # 이제 잘됨
print(W2.eval())
# 텐서에서variable 함수 선언시 꼭 global_variables_initializer()호출해야 사용가능

#tf.placeholder 와 feed_dict는 입력 데이터를 넣기 위해 사용
input1 = tf.placeholder(tf.float32,3) # tf.float32 형 값을 3개 가지는 placeholder
input2 = tf.placeholder(tf.float32)# tf.float32 형 값을 가지는 placeholder
#placeholder = 변수 /구체적인 형태가 값이 정해지지 않은 임시 변수
#feed_dict = 파이썬의 딕셔너리 형 / placeholder와 실제 값을 연결하는 견결 하는 역할

output = tf.multiply(input1,input2)
session = tf.InteractiveSession()

print(session.run([output],feed_dict={input1:[1.,2.,3.], input2:[3.]}))


a = tf.constant(2) # 상수 텐서 생성
b = tf.constant(3)
x = tf.add(a,b) # 텐서끼리 덧셈 연산
writter = tf.summary.FileWriter('./graphs', session.graph)
#현재 폴더 아래의 그래픽 폴더에 현재 세션의 그래프 정보를 저장
session.run(x) #실행 2+3 = 5 가 나옴
writter.close()
session.close()

session = tf.InteractiveSession()
a = tf.constant(2, name='a')
b = tf.constant(3, name='b')

x = tf.add(a,b)
writer = tf.summary.FileWriter('./graphs', session.graph)

session.run(x)
writer.close()
session.close()


import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

boston = datasets.load_boston()
boston_slice = [x[5] for x in boston.data]#6번째 피처만 사용

#텐서플로에서 사용할 크기로 조정
#reshape 함수로 데이터를 열이 1인 데이터형을 변환
boston_slice
data_x = np.array(boston_slice).reshape(-1,1)
data_y = boston.target.reshape(-1,1)
print(data_x.shape, data_y.shape)
# 크기가 506 행 1열 인 행렬 생성(데이터안의 총 샘플수 X 피처수)
#선형 회귀는 y = wx + b인 선형 행렬

n_sample = data_x.shape[0] #data_x의 열 개수
x = tf.placeholder(tf.float32, shape = (n_sample,1),name = 'X')

y = tf.placeholder(tf.float32, shape = (n_sample,1),name = 'y')

W = tf.Variable(tf.zeros((1,1)), name='weights')

b = tf.Variable(tf.zeros((1,1)), name = 'bias')
# 여기까지 기본적인 변수를 정의함

#학습함수, 손실함수, 최적화 함수를 정해야됨.
# 학습함수 : 입력데이터 X 기울기 + 편향
# 손실함수 : (타깃값 - (입력데이터 X 기울기 + 편향) ) 제곱 / 샘플 수
#


matmul(a, b, transpose_a=False, transpose_a=False, adjoin_a=False,
       adjoin_b=False, a_is_sparse=False, b_is_sparse=False, name=None)
#함수의 원형
#a 와 b는 같은 데이터형의 요소를 가져야함
#a 와 b는 같은 행렬 개수를 가져야함

square(x, name=None)
#X 입력 텐서. 하나의 형을 가짐. name 연산명

reduce_mean(input_tensor, axis=None, keep_dims=False, name=None,
            reduction_indices=None)
#input 입력텐서, asix 줄일 차원 수, keep_dims True 면 줄인 차원의 인덱스를 기억
#name 연산명, reducation_indices : axis 의 옛 이름(현재 사용 X)


#최적화 방법: 경사하강법
tf.train.GradientDescentOptimizer.__init__(learing_rate, use_locking=False,
                                           name='GradientDescent')
#learing_rate 는 학습률
#텐서플로우는 경사하강법 외에도 AdaGrad Adam등의 최적화 클래스를 제공

y_pred = tf.matmul(x,W) + b
loss = tf.reduce_mean(tf.square(y_pred - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss) # 최적화 함수. 손실함수의 최솟값을 찾음
summary_op = tf.summary.scalar('loss',loss)

def plot_graph(y, fout):
    plt.scatter(data_x.reshape(1,-1)[0], boston.target.reshape(1,-1)[0])
    plt.plot(data_x.reshape(1,-1)[0],y.reshape(1,-1)[0])
    plt.savefig(fout)
    plt.clf()


with tf.Session() as sess:
    # sess = tf.InteractiveSession() 직접 할때
    sess.run(tf.global_variables_initializer())#변수 초기화된
    summary_writer = tf.summary.FileWriter('./graphs', sess.graph)

    #텐서보드를 위해 지정한 그래프를 이용해 생성
    y_pred_before = sess.run(y_pred, {x: data_x})

    #학습 전의 예측된 기울기 상태
    plot_graph(y_pred_before, 'before.png')
    for i in range(100):
        #loss 연산 , summary_op연산, train_op 연산을 수행
        #loss 연산의 결과를 loss_t, summary_op 연사의 결과를 summary에 받음
        loss_t, summary, _ = sess.run([loss, summary_op, train_op],
                                      feed_dict={x: data_x, y: data_y})

        summary_writer.add_summary(summary,i)

        #if i%10 == 0:
        print('loss = % 4.4f' % loss_t.mean())
        y_pred_after = sess.run(y_pred, {x : data_x})
        # plot_graph(y_pred_after,'after'+ str(i) +'.png')


    y_pred_after = sess.run(y_pred,{x:data_x})
    plot_graph(y_pred_after,'after.png')
    summary_writer.close()

import tensorflow as tf
tf.test.is_gpu_available()
