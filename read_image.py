
import tensorflow as tf
import matplotlib.image as mp_img
import matplotlib.pyplot as plot
import os

filename="./colorful_image.jpeg"
image = mp_img.imread(filename)

print ("Original image shape: ", image.shape)

x = tf.Variable(image, name='x')

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    transpose = tf.image.transpose_image(x)
    result = sess.run(transpose)

    print ("Transpose: ", result.shape)
    sess.close()
