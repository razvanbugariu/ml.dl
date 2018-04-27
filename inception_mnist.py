import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data as mnist_data

mnist = mnist_data.read_data_sets('MNIST_data', one_hot=True)

MEAN = np.mean(mnist.train.images)
STD = np.std(mnist.train.images)

def resize_images(images):
    reshaped = (images - MEAN) / STD
    reshaped = np.reshape(reshaped, [-1, 28, 28 ,1])

    return reshaped

def nielse_net(inputs, is_training, scope='NielsenNet'):

    with tf.variable_scope(scope, 'NielsenNet'):
        net = slim.conv2d(inputs, 20, [5, 5], padding='SAME', scope='layer1-conv')
        net = slim.max_pool2d(net, 2, stride=2, scope='layer1-max-pool')

        net = slim.conv2d(net, 40, [5, 5], padding='SAME', scope='layer2-conv')
        net = slim.max_pool2d(net, 2, stride=2, scope='layer2-max-pool')

        net = tf.reshape(net, [-1, 5*5*40])

        net = slim.fully_connected(net, 1000, scope='layer5')
        net = slim.dropout(net, is_training=is_training, scope='layer5-dropout')

        net = slim.fully_connected(net, 1000, scope='layer6')
        net = slim.dropout(net, is_training=is_training, scope='layer6-dropout')

        net = slim.fully_connected(net, 10, scope='output')
        net = slim.dropout(net, is_training=is_training, scope='output-dropout')

        return net

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='Inputs')
y_actual = tf.placeholder(tf.float32, shape=[None, 10], name='Labels')
is_training=tf.placeholder(tf.bool, name='IsTraining')

logits = nielse_net(x, is_training, scope='NielsenNetTrain')

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_actual, logits=logits))
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_step = tf.train.MomentumOptimizer(0.01, 0.5).minimize(cross_entropy)

loss_summary = tf.summary.scalar('loss', cross_entropy)
accuracy_summary = tf.summary.scalar('accuracy', accuracy)

sess.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter('/tmp/nielsen-net', sess.graph)

eval_data = {
    x: resize_images(mnist.validation.images),
    y_actual: mnist.validation.labels,
    is_training: False
}

for i in range(100000):
    images, labels = mnist.train.next_batch(100)
    summary, _ = sess.run([loss_summary, train_step], feed_dict={x: resize_images(images), y_actual: labels, is_training: True})
    train_writer.add_summary(summary, i)

    if i % 1000 == 0 :
        summary, acc = sess.run([accuracy_summary, accuracy], feed_dict=eval_data)
        train_writer.add_summary(summary, i)
        print ("Step: %5d, Validation accuracy = %5.2f%%" % (i, acc * 100))
