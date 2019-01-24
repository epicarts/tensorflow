import tensorflow as tf


a = tf.constant(20, name='a_node')
b = tf.constant(40, name='b_node')

sess = tf.Session()

writer = tf.summary.FileWriter('.\\testgraph', sess.graph)

sess.run(a)
sess.run(b)
