import tensorflow as tf  # Graph Input
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

# model weight
# 1,2 2차원 배열
W = tf.Variable(tf.random_uniform([1, len(x_data)], -5.0, 5.0))

# Construct a linear model
hypothesis = tf.matmul(W, x_data)

# Cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# Minimize
a = tf.Variable(0.1)  # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

# Launch the graphs
sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W))
