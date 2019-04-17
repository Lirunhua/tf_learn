# -*- coding: utf-8 -*-

import tensorflow as tf

# if not assign a graph, all variables and operation will be defined in default graph
g1 = tf.Graph()
g2 = tf.Graph()

g2.as_default()
# g1.as_default()
# assert g1 == tf.get_default_graph()
# assign g1 as the default graph
with g1.as_default():
    w1 = tf.placeholder(float, name='w1')
    w2 = tf.placeholder(float, name='w2')
    b1 = tf.Variable(2.0, name='b1')
    feed_dict = {w1: 4, w2: 8}
    w3 = tf.add(w1, w2, name='add')
    w4 = tf.multiply(w3, b1, name="op_to_multiply")

with g2.as_default():
    b3 = tf.Variable(6.0, name='b3')
    b4 = tf.Variable(9.0, name='b4')
    b5 = tf.Variable(2.80, name='b5')
    b6 = tf.Variable(2.90, name='b6')
    b7 = tf.Variable(2.80, name='b7')
    w9 = tf.add(b3, b6, name='add_g2')

with tf.Session(graph=g2) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w9))
    print([ops.name for ops in sess.graph.get_operations()])

with tf.Session(graph=g1) as sess:
    # what tf do when tf.global_variables_initializer()
    # all 
    sess.run(tf.global_variables_initializer())
    feed_defalut = {w1: 1.2, w2: 1.4}
    print("result: ", sess.run(w4, feed_dict=feed_defalut))
    ops = [op.name for op in sess.graph.get_operations()]
    print("graphs in g1\n")
    print(ops)
    multiply_tensor = sess.graph.get_tensor_by_name('op_to_multiply:0')
    add_tensor = sess.graph.get_tensor_by_name('add:0')
    print(sess.run(multiply_tensor, feed_dict={w1: 1.2, w2: 1.4, add_tensor: 100}))
#    print('multiply_tensor: ', multiply_tensor.eval())


# Graph of w4 will be the default graph
print("graph of w4: ", w4.graph)
print("Whether the graph of w4 is the default graph: ", w4.graph is tf.get_default_graph())

# Define a new graph
g1 = tf.Graph()
with g1.as_default():
    # Define variable c in g2
    c = tf.get_variable("c", initializer=tf.zeros_initializer, shape=(1))

# define the second graph
g2 = tf.Graph()
# Set the g2 as the default graph
with g2.as_default():
    # Define variable C in graph g2, and initiate c as 1
    c = tf.get_variable("c", initializer=tf.ones_initializer, shape=(1))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Create a saver object which will save all the variables


with tf.Session() as sess:
    w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
    w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print("save the model ")
    saver.save(sess, './my_test_model/', global_step=100)
