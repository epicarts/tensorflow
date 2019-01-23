'''
https://www.youtube.com/watch?v=eBbEDRsCmv4&t=1105s
텐서보드를 활용한
'''

import tensorflow as tf

a = tf.constant([1])
writer = tf.summary.FileWriter("file_path")
writer.add_graph(sess.graph)
