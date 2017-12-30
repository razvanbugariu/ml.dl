import tensorflow as tf

sess = tf.Session()

W = tf.Variable([0.0001], dtype=tf.float32)
b = tf.Variable([0.01], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = tf.add(tf.multiply(W, x), b)

init = tf.global_variables_initializer()
sess.run(init)
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(1000):
    sess.run(train, {x: [1, 2, 3 , 4], y:[-1, -2, -3 , -4]})

print (sess.run([W, b]))
