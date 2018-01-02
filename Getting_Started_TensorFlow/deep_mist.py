import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W1 = tf.get_variable("W1", shape=[784, 200],
           initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.zeros([200]))

W2 = tf.get_variable("W2", shape=[200, 100],
           initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.zeros([100]))

W3 = tf.get_variable("W3", shape=[100, 10],
           initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

Z1 = tf.add(tf.matmul(x, W1), b1)
A1 = tf.nn.relu(Z1)
Z2 = tf.add(tf.matmul(A1, W2), b2)
A2 = tf.nn.tanh(Z2)
y = tf.add(tf.matmul(A2, W3), b3)

# or loss
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels= y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range (1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict = {x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
