import tensorflow as tf


a = tf.constant(20, name='a_node')
b = tf.constant(40, name='b_node')

c = tf.add(a, b)
sess = tf.Session()

dir(c.op)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('.\\testgraph', sess.graph)
    print(sess.run(a))
    sess.run(b)
    sess.run(c)
    writer.close()
