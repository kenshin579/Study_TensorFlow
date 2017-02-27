import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis
hypothesis = W * X

# Simplified const function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# minimize
lr = 0.1
descent = W - tf.multiply(lr, tf.reduce_mean(tf.multiply((tf.multiply(W, X) - Y), X)))
train = W.assign(descent)

# Before starting, initialize the variables
init = tf.global_variables_initializer()

# Launch the graph
sess = tf.Session()
sess.run(init)

# Fit the line
for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

print(sess.run(hypothesis, feed_dict={X: 5}))
print(sess.run(hypothesis, feed_dict={X: 2.5}))