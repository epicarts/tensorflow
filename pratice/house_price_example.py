import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from pandas import Series, DataFrame


#집 평수와 가격
df = DataFrame({'size': [2104, 1416, 1534, 852],
                'price': [460, 232, 315, 178]})
df
#평수가 커질수록 가격은 올라갈 것이다.
#집값을 예측 해보쟈.
theta_1
theta_2

size = tf.placeholder(tf.float32)
price = tf.placeholder(tf.float32)
W = tf.Variable([1.], tf.float32, name='Weight')
b = tf.Variable([1.], tf.float32, name='bais')

#집값 X 가중치 + b = 평수 theta1 = W, theta2 = b
h = W*price + b

cost = tf.reduce_mean(tf.square(h - size))

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='GD')
train = optimizer.minimize(cost)

'''
가설을 단순하게 바꾸어보쟈.
만약 b를 0으로 고정 시킨다면 h는 원점을 지나는 직선이 될 것이다.
'''
train_data = DataFrame({'size': [1, 2, 3, 4],
                        'price': [1, 2, 3, 4]})
train_data['size'][0]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(len(train_data['size'])):
        _, cost_val, h_val = sess.run([train, cost, h],
                                      feed_dict={size:train_data['size'][i], price:0})
        print(cost_val, h_val)
        print(i)

plt.scatter(train_data['size'],train_data['price'])
plt.plot(train_data['size'],func_y(train_data['size'], 0.5))
plt.plot(train_data['size'],func_y(train_data['size'], 1),'r')

func_y(train_data['size'])
