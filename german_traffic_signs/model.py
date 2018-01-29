import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
from tensorflow.python.client import device_lib
from skimage import data as skimage_data
from skimage import transform
from skimage.color import rgb2gray

plt.style.use('ggplot')

def get_devices():
    local_devices_protos = device_lib.list_local_devices()
    return [x.name for x in local_devices_protos]

def load_data(data_directory):
    # directories = [];
    # for d in os.listdir(data_directory):
    #     if os.path.isdir(os.path.join(data_directory, d)):
    #         directories.append(d)
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    images = []
    labels = []

    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = []
        for f in os.listdir(os.path.join(data_directory, d)):
            if(f.endswith(".ppm")):
                file_names.append(f)

        for f in file_names:
            images.append(skimage_data.imread(os.path.join(label_directory,f)))
            labels.append(int(d))

    return images, labels

def readFromCSV(directoryPath):
    file = open(directoryPath + "/GT-final_test.csv", "rt")
    reader = csv.reader(file)
    file_names = []
    labels = []
    images = []
    for row in reader:
        values = " ".join(row)
        file_names.append(values.split(";")[0])
        labels.append(values.split(";")[7])
    file.close()
    file_names.remove("Filename")
    labels.remove("ClassId")
    for f in file_names:
        images.append(skimage_data.imread(os.path.join(directoryPath,f)))
    return images, labels

print ("Reading training data...")
images_train, labels_train = load_data("/home/albatros/Workspace/MachineLearning/german_traffic_signs/GTSRB/Final_Training/Images")
print ("Done reading training data")
print ("Reading test data...")
images_test, labels_test = readFromCSV("/home/albatros/Workspace/MachineLearning/german_traffic_signs/GTSRB/Final_Test/Images")
print ("Done reading test data")

images_train = [transform.resize(image, (28,28), mode="constant") for image in images_train]
images_train = np.asarray(images_train)

images_test = [transform.resize(image, (28,28), mode="constant") for image in images_test]
images_test = np.asarray(images_test)

print ("Trainingset:")
print ("img ndim: %d" % (images_train.ndim))
print ("number of images %d" % (len(images_train)))

print ("number of labels: %d" % (len(labels_train)))
print ("unique labels: %d" % (len(set(labels_train))))

print ("image shape", images_train[0].shape)

print ("\n")

print ("Testset:")
print ("img ndim: %d" % (images_test.ndim))
print ("number of images %d" % (len(images_test)))

print ("number of labels: %d" % (len(labels_test)))
print ("unique labels: %d" % (len(set(labels_test))))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev= 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.001, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1], strides = [1, 2, 2, 1],
                                padding = 'SAME')

x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28, 3])
y = tf.placeholder(dtype = tf.int32, shape = [None])
keep_prob = tf.placeholder(tf.float32)

#Input Layer
with tf.name_scope('input-layer') as scope:
    input_layer = tf.cast(x, tf.float32)

with tf.name_scope('conv1_layer') as scope:
    W_conv1 = weight_variable([5, 5, 3, 64])
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(input_layer, W_conv1) + b_conv1)

    with tf.name_scope('pooling') as scope:
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('dropout') as scope:
        pool1_drop = tf.nn.dropout(h_pool1, keep_prob)

with tf.name_scope('fc1_layer') as scope:
    with tf.name_scope('flatten') as scope:
        pool1_flat = tf.reshape(pool1_drop, [-1, 14*14*64])

    W_fc1 = weight_variable([14*14*64, 128])
    b_fc1 = bias_variable([128])
    h_fc1 = tf.nn.relu(tf.matmul(pool1_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout') as scope:
        fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('fc2_layer') as scope:
    W_fc2 = weight_variable([128, 64])
    b_fc2 = bias_variable([64])

with tf.name_scope('logits') as scope:
    logits = tf.matmul(fc1_drop, W_fc2) + b_fc2

loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels = y,
        logits = logits))

train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess = tf.Session()

sess.run(tf.global_variables_initializer())
for i in range(201):
    _, loss_value = sess.run([train_op, loss], feed_dict = {x: images_train, y: labels_train, keep_prob:0.5})
    if i % 10:
        print ("Loss: ", loss_value)
