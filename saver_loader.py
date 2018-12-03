# -*- coding: utf-8 -*-

import tensorflow as tf

# All variables and operation will be defined in default graph
w1 = tf.placeholder(float, name='w1')
w2 = tf.placeholder(float, name='w2')
b1 = tf.Variable(2.0, name='b1')
feed_dict = {w1: 4, w2: 8}

w3 = tf.add(w1, w2)
w4 = tf.multiply(w3, b1, name="op_to_restore")

# Graph of w4 will be the default graph
print("graph of w4: ", w4.graph())
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
    c = tf.get_variable("c", initializer=tf.ones_initializer,shape=(1))

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
