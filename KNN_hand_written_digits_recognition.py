#Using K-nearest-neighbors algorithm to recognize hand written digits from
# MNIST dataset
#Steps:
# 1.Getting MNIST images training and test data. These images will be separated
# in batches using Tensorflow libraries
# 2.Calculating L1 distance
# 3.Running the algorithm. Predict labesl for all the test data and measure
# accuracy
import numpy as np
import tensorflow as tf

#Import mnist data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

#retrieve training digits and labels into batches of size 4096
training_digits, training_labels = mnist.train.next_batch(4096)
#the same for test
test_digits, test_labels = mnist.test.next_batch(256)

#784 = height * width of image, this is gray scale so you do not have to * 3
#(as you did with color images)
# None = array o images.
training_digits_pl = tf.placeholder(tf.float64, shape = [None, 784])
test_digits_pl = tf.placeholder(tf.float64, shape = [784])

#Calculate L1 between test image and the entire training set_shape
# absolute value of x - y
l1_distance = tf.abs(tf.add(training_digits_pl, tf.negative(test_digits_pl)))

#calculate the sum on rows. the closest to 0 if the most nearest neighbor
distance = tf.reduce_sum(l1_distance, axis = 1)

#find which index has the most closest to 0 sum
pred = tf.arg_min(distance, 0)

#calculating the accuracy

accuracy = 0

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(len(test_digits)):
        #get the nearest neighbor for every element in test set against the entire training set.
        nn_index = sess.run(pred,
        feed_dict={training_digits_pl: training_digits, test_digits_pl: test_digits[i, :]} )

        print ("Test", i, "Prediction:", np.argmax(training_labels[nn_index]),
        "True Label:", np.argmax(test_labels[i]))

        if np.argmax(training_labels[nn_index]) == np.argmax(test_labels[i]):
            accuracy += 1/len(test_digits)

    print ("Done")
    print ("Accuracy:", accuracy)
    sess.close()
