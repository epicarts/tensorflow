import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import pandas as pd
from pandas import DataFrame, Series

'''
3d 그래프를 텐서플로우를 사용하여 그려보쟈!!
1. 3d 그래프에 대한 설정을 한다.(x좌표: Weight, y좌표: Bais, z좌표: cost)
2. x좌표와 y좌표를 전부 이용한다. cost 함수는 모든 x, y좌표값을 사용함
3. 모든 weight 와 bais를 cost 함수에 대입한 모든 값을 구해야한다.
4. weight값이 고정되어있을때, 혹은 bais 값이 고정되어있을때 cost를 생각해보라.
5. 위 식을 토대로 텐서플로우를 활용해서 그려보자.
'''
#17*x^2-16*abs(x)*y+17*y^2=225
#z = func_z(x, y)#z = x**2/5. + x*y/50. + y**2/5.


w_range = np.arange(-2.0, 4.0, 0.05)
b_range = np.arange(-2.0, 4.0, 0.05)
w, b = np.meshgrid(w_range, b_range)
x = [1., 2., 3., 4., 5.]
y = [2.1, 3.1, 4.1, 5.1, 6.1]
w.shape
b.shape
cost_graph_3d(w, b, x, y)
z = cost_graph(w, b, x)
epoch = 10
w = 3
b = 3
learning_rate=0.05
def cost_graph_3d_gradient(w, b, x, y, epoch = 5, learning_rate=0.05, figsize=(8, 5)):
    '''
    3차원 공간에서 gradients를 사용하여 최저점을 찾아간다.
    w: 랜덤으로 주어진 weight 시작 값
    b: 랜덤으로 주어진 bais 시작 값
    x: input data [1., 2., 3., 4., 5.]
    y: label data [1., 2., 3., 4., 5.]
    '''

    #초기 weight 값 설정하기.
    weight_init = tf.constant([w], tf.float32)

    #초기 bais값 설정하기.
    bais_init = tf.constant([b], tf.float32)

    #Variable()생성자로 tensor를 전달 받음. constant or random
    weight = tf.Variable(weight_init, name="weights")
    bais = tf.Variable(bais_init, name='bais')

    x_data = tf.constant(x, tf.float32)
    y_data = tf.constant(y, tf.float32)
    x_data

    #가설 설정
    hypothesis = weight * x_data + bais
    hypothesis

    #cost 설정 / 최소 제곱 오차법
    cost = tf.reduce_mean((tf.square(hypothesis - y_data)))

    #GradientDescentOptimizer를 사용하지 않고 tf.gradients 를 사용할 것이다.
    d_bais, d_weight = tf.gradients(cost, [bais, weight])

    #기존값 - 미분값*learning_rate
    #assign 값을 바꿈. 기존 값 ==> 계산한 식으로 변경
    update_w = tf.assign(weight, weight - (learning_rate * d_weight))
    update_b = tf.assign(bais, bais - (learning_rate * d_bais))

    #모델 파라미터를 업데이트 하는 텐서를 실행
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #시작 위치 저장
        b_result = np.array([b], np.float32)
        w_result = np.array([w], np.float32)
        c_result = np.array([sess.run(cost)])
        #모든 데이터를 돌리는 횟수(epoch)
        for step in range(epoch):
            #update를 하여 기존 assign을 변경 시킨다.
            sess.run([update_b, update_w])

            #값이 바뀔때 마다 배열에 저장
            b_val, w_val, c_val = sess.run([bais, weight, cost])
            b_result = np.append(b_result, b_val)
            w_result = np.append(w_result, w_val)
            c_result = np.append(c_result, c_val)

    x
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.contour(w_result, b_result, c_result, levels=np.logspace(0, 5, 35), cmap=plt.cm.jet)

    '''
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': '3d'})
    #그래프의 범위는 -|최대값| ~ |최대값|
    #x축 설정
    w_abs = abs(w_result.max())
    ax.set_xlabel('Weight')
    ax.set_xlim([-w_abs, w_abs])

    #y축 설정
    b_abs = abs(b_result.max())
    ax.set_ylabel('Bais')
    ax.set_ylim([-b_abs, b_abs])

    #cost 범위는 최소 0 ~ |최대값|
    #z축 설정
    c_abs = abs(c_result.max())
    ax.set_zlabel('Cost')
    ax.set_zlim([0, c_abs])

    ax.invert_xaxis()
    ax.plot_surface(w, b, z, cmap='jet',rstride=1, cstride=1, linewidth=0)
    print("minimum cost:", z.min())
    plt.show()
    '''
    w_result.shape
    b_result.shape
    c_result.shape
    #여기서 움직이는 그래프 그리기.
    #화살표 추가
    #3d 범위를 구해야함.


    #gradients를 사용하여 나온 b의 범위와 w의 범위를 이용하여, w, b값을 대입하쟈
    #그래야 그래프 그리기 쉬울 듯.
    cost_graph_3d(w, b, x, y, figsize=(8, 5))

def two_variable_graph_gradient():
    '''
    두 변수를 그린 그래프에서 최저 값을 찾아 내려가는 그래프를 그릴 예정ㄴ

    '''
    x_gd, y_gd, z_gd = gradient_descent(x0, y0, learning_rate, epoch)
    ax1.plot(x_gd, y_gd, 'bo')

    for i in range(1, epoch+1):
        ax1.annotate('', xy=(x_gd[i], y_gd[i]), xytext=(x_gd[i-1], y_gd[i-1]),
                       arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                       va='center', ha='center')


def two_variable_graph():
    '''
    두 변수의 상관관계를 나타낸 그래프를 그릴예정
    여기에선 Weight 값과 bais 값사이를 알수 있겟징..
    '''
    a = np.arange(-3, 3, 0.05)
    b = np.arange(-3, 3, 0.05)

    x, y = np.meshgrid(a, b)
    z = func_z(x, y)

    fig1, ax1 = plt.subplots()

    ax1.contour(x, y, z, levels=np.logspace(-3, 3, 25), cmap='jet')


def cost_graph_3d(w, b, x, y, figsize=(8, 5)):
    '''
    w: weight 범위 np.meshgrid() 결과 값.
    b: bais 범위   np.meshgrid() 결과 값.
    x: input data [1., 2., 3., 4., 5.]
    y: label data [1., 2., 3., 4., 5.]
    데이터 X, 예측값 Y, cost 함수(tensor type)
    '''

    #Weight와 bais 3차원으로 변환, (x, y, 1)
    weight = tf.constant(w, shape=(w.shape[0], w.shape[1],1),dtype=tf.float32)
    weight
    bais = tf.constant(b, shape=(w.shape[0], w.shape[1],1),dtype=tf.float32)
    bais

    #차원 변경 (1, 1, data size)
    x_data = tf.constant(x,shape=(1, 1, len(x)),dtype=tf.float32)
    x_data
    y_data = tf.constant(y,shape=(1, 1, len(y)),dtype=tf.float32)
    y_data

    ##z = x**2/5. + x*y/50. + y**2/5.
    #가설 만들기 y = ax + b
    hypothesis = weight * x_data + bais
    hypothesis

    #cost 함수  3차원 값 합치기((예측값 - 실제값)^2 /5)
    cost = tf.reduce_mean(tf.square(hypothesis - y_data), axis=2)
    cost

    with tf.Session() as sess:
        z = sess.run(cost)

    #여기서 부터 그래프 그리기
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': '3d'})

    #x축 설정
    ax.set_xlabel('Weight')
    ax.set_xlim([w.min(), w.max()])

    #y축 설정
    ax.set_ylabel('Bais')
    ax.set_ylim([b.min(), b.max()])

    #z축 설정
    ax.set_zlabel('Cost')
    ax.set_zlim([z.min(), z.max()])

    ax.invert_xaxis()
    ax.plot_surface(w, b, z, cmap='jet',rstride=1, cstride=1, linewidth=0)
    print("minimum cost:", z.min())
    plt.show()

    fig, ax2 = plt.subplots(figsize=(10, 6))
    ax2.contour(w, b, z, cmap='jet',norm=LogNorm())
a = LogNorm()
a
from matplotlib.colors import LogNorm

cost_graph_3d(w, b, x, y)
