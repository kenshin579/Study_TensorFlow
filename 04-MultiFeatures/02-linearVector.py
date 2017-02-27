import tensorflow as tf

# Graph Input
x_data = [[0., 2., 0., 4., 0.],
          [1., 0., 3., 0., 5.]]
y_data = [1, 2, 3, 4, 5]

# model weight
# 1,2 2차원 배열
W = tf.Variable(tf.random_uniform([1,2], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# Construct a linear model
hypothesis = tf.matmul(W, x_data) + b

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
        print(step, sess.run(cost), sess.run(W), sess.run(b))
