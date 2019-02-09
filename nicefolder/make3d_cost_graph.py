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

data.X
data = DataFrame({'X': [1., 2., 3., 4., 5.], 'Y': [-14, -21, -28, -35, -42]},dtype=np.float32)

x = np.arange(-10.0, 10.0, 0.1)
y = np.arange(-10.0, 10.0, 0.1)
xx, yy = np.meshgrid(x, y)
len(data.X)
cost_graph_3d(xx,yy,data.X, data.Y)

def cost_graph_3d(w, b, x, y):
    '''
    w: weight
    b: bais
    x: input data
    y: label data
    데이터 X, 예측값 Y, cost 함수(tensor type)
    '''
    bais = tf.constant(ax_YY, tf.float32)
    weight = tf.constant(xx,shape=(None, None,len(x)),dtype=tf.float32)
    weight


    weight
    weight
    x_data = tf.constant(data.X, tf.float32)
    y_data = tf.constant(tf.shape(1,1,?), tf.float32)

    #가설 만들기
    hypothesis = weight * x_data + bais



    #여기서 부터 그래프 그리기
    fig, ax = plt.subplots(figsize=(8, 8),subplot_kw={'projection': '3d'})

    #x축 설정
    ax.set_xlabel('Weight')
    ax.set_xlim([x.min(), x.max()])

    #y축 설정
    ax.set_ylabel('Bais')
    ax.set_ylim([y.min(), y.max()])

    #z축 설정
    ax.set_zlabel('Cost')
    ax.set_zlim([zz.min(), zz.max()])


'''
'''
'''
'''
#임의로 데이터 생성
x = np.arange(1.0, 5, 0.1)
y = np.arange(1.1, 5, 0.1)

xx, yy = np.meshgrid(x, y)

weight = tf.constant(xx, tf.float32)
bais = tf.constant(yy, tf.float32)
xx.shape
xxx = xx.reshape(200,200,1)
yy.shape
yyy = yy.reshape(200,200, 1)

xxx.shape
c = np.array(data.X).reshape(1,1,5)
c.shape

xxx[0][0]
c[0][0]

z = xxx * c
z.shape
z[0][0]
yyy.shape
hhh = z + yyy
hhh.shape

asd = np.array(data.Y).reshape(1,1,5)

zzz = np.square(hhh - asd)/5 #전부 제곱하고 5로 나눔.
zzz.shape
zzz[0][7]

zz = zzz.sum(axis=2)
zz.shape
zz[0][0]

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
#x축 설정
ax.set_xlabel('Weight')
ax.set_xlim([x.min(), x.max()])

#y축 설정
ax.set_ylabel('Bais')
ax.set_ylim([y.min(), y.max()])

#z축 설정
ax.set_zlabel('Cost')
ax.set_zlim([zz.min(), zz.max()])
ax.invert_xaxis()
ax.plot_surface(xx, yy, zz, linewidth=0)
zz.shape
zzz[0][0]
zzz[0][1]
zzz[0][2]
zzz[0][3]
