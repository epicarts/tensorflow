import tensorflow as tf
import numpy as np

'''
tf.graidents(y, x) ~ dy/dx
첫번째는 편미분을 하려는 대상의 텐서
두번째는 편미분 변수에 해당하는 텐서
'''
x = tf.constant(1.0)
b = tf.constant(2.0)

y = 2*x + b

#x에 대해서 편미분을 하겠다.
with tf.Session() as sess:
    result = sess.run(tf.gradients(y, x)) # dy/dx
result

#b에 대해서 편미분을 하겠다.
with tf.Session() as sess:
    result = sess.run(tf.gradients(y, b)) # dy/dx
result

#시그모이드를 편미분 하겠다.
sigmoid = tf.sigmoid(y)
with tf.Session() as sess:
    #y에 대해서 미분함. 오직 시그모이드를 미분한 값.
    result = sess.run(tf.gradients(sigmoid, y))
result#0.01766

#순수하게 시그모이드를 미분
sigmoid_z = 1/(1+np.exp(-4.0))
sigmoid_z * (1 - sigmoid_z)#0.01766


#x에 대한 시그모이드를 편미분 하겠다. 미분의 체인룰 적용.
sigmoid = tf.sigmoid(y)
with tf.Session() as sess:
    result = sess.run(tf.gradients(sigmoid, x))
result#0.0353

#ds/dx = ds/dy * dy/dx 로 나눌수 있음. dy를 추가 시킴으로써
#ds/dy = tf.gradients(sigmoid, y) = 0.01766
#dy/dx = tf.gradients(y, x) = 2.0
with tf.Session() as sess:
    #sess.run(tf.gradients(sigmoid, x)) 같은 식임
    result = sess.run(tf.gradients(sigmoid, y)) + sess.run(tf.gradients(y, x))
result[0] * result[1]


#렐루(ReLU) 함수를 편미분 하겠다.
#y > 0 미분값 1
r = tf.nn.relu(y)
with tf.Session() as sess:
    result = sess.run(tf.gradients(r, x))
result

#y <= 0 미분값 0
r = tf.nn.relu(y - 5)
with tf.Session() as sess:
    result = sess.run(tf.gradients(r, x))
result

#기존에 사용했던 경사하강법 조절
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(cost)

#새로 변경된 경사 하강법.
#모델 W, b에 대해 손실 함수 cost 식의 미분을 구함
dW, db = tf.gradients(cost, [W, b])
#업데이트된 값 = 기존값 - 학습률 * 미분한 값
update_W = tf.assign(W, W - 0.5 * dW)
update_b = tf.assign(b, b - 0.5 * db)
