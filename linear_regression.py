import tensorflow as tf

#Model parameters. W and b, tensorflow variables with values [.3] and respectively [-.3].
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

x = tf.placeholder(tf.float32)
linear_regressin = tf.add(tf.multiply(W, x), b)

y = tf.placeholder(tf.float32)

loss = tf.reduce_sum(tf.square(linear_regressin - y))

optimizer = tf.train.GradientDescentOptimizer(0.01)

train = optimizer.minimize(loss)

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(10000):
        sess.run(train, {x: x_train, y: y_train})

    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})

    print("W: %s\nb: %s\nloss: %s"%(curr_W, curr_b, curr_loss))
    sess.close()
