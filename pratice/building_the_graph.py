import tensorflow as tf

c = tf.constant(4.0)
assert c.graph is tf.get_default_graph()

g = tf.Graph()
with g.as_default():
  # Define operations and tensors in `g`.
  c = tf.constant(30.0)
  assert c.graph is g

tf.Graph.as_default()

# 1. Using Graph.as_default():
g = tf.Graph()

#g를만들고 g안에 있는 graph.as_default사용
with g.as_default():
  c = tf.constant(5.0)
  assert c.graph is g

g
c.graph

# 2. Constructing and making default:
with tf.Graph().as_default() as g:
  c = tf.constant(5.0)
  assert c.graph is g

 tf.Graph().as_default()
g
