import tensorflow as tf



node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sess = tf.Session()
#sess.run(op)
sess.run(node3)


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b
#sess.run(op, feed_dict={x: x_data})
sess.run(adder_node, feed_dict={a: 3, b:4.5})
sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]})#여러개 값도 전달 가능



writer = tf.summary.FileWriter("./testgraph/")
writer.add_graph(sess.graph)  # Show the graph

import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame, Series



exam = pd.DataFrame({'hours': [10., 9., 3., 2.], 'score': [90., 80., 50., 30.]})
exam

'''
데이터 분석을 해보쟝
'''
fig = plt.plot()
plt.plot(exam.hours, exam.score, marker='o', linestyle='', ms=15)
'''
hours score 비례할 것이다.
공부한 시간과 시험 점수는 비례할 것이다 라고 가정
y = ax + b
인풋은 시간, 내가 예상하는건  score
'''
x_train = exam.hours.values.tolist()
y_train = exam.score.values.tolist()

#텐서플로우가 사용하는 변수.
#trainable Variable 라고 생각가능.
b = tf.Variable(tf.zeros([1]), tf.float32, name='bais')#값이 1개인 1차원 array
W = tf.Variable(tf.random_normal([1]), name='x_data')

pred_y = x_train*W + b
pred_y
#비용, 손실율. 을 이정도로 하겠다.
cost = tf.sqrt(tf.reduce_mean((tf.square(pred_y - y_train))))

#조절은 어떻게 할 것인가 ??
#조절은 경사하강법을 하겠다.
learing_rate = 0.03
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learing_rate)

#cost를 최소화 하겠다.
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100000):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
    if sess.run(cost) < 3.0:
        break
writer = tf.summary.FileWriter("./testgraph/")
writer.add_graph(sess.graph)  # Show the graph


plt.plot(x_train,y_train,marker='o',linestyle='', ms=15)
plt.plot(x_train ,sess.run(pred_y))
