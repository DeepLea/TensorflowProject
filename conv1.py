#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)



import tensorflow as tf



import tensorflow.contrib.learn as skflow
from sklearn.datasets import fetch_lfw_people
import numpy as np
import tensorflow as tf
lfw_people = fetch_lfw_people(min_faces_per_person = 70, resize = 0.88, funneled = False,color = False, slice_ = (slice(0,250),slice(0,250)))  # Det har tar tid, upp till 2 min
from random import random
INPUT_SIZE = lfw_people.data.shape[1]
OUTPUT_SIZE = len(lfw_people.target_names)-1 #6 # There are > 6000 unique persons: 2^13 > 6000
LENGTH = len(lfw_people.data)
print(LENGTH)
print(OUTPUT_SIZE)
print(len(lfw_people.target_names)-1)
print(len(lfw_people.images[0]))
print(len(lfw_people.images[0][0]))


###############################################################################
#      Saving the targets as binary in each row
# y_ = np.zeros((lfw_people.target.shape[0] ,OUTPUT_SIZE), dtype=np.int)
# get_bin = lambda x, n: format(x, 'b').zfill(n)
# for i, t in enumerate(lfw_people.target):
#   for j, n in enumerate(get_bin(lfw_people.target[i], OUTPUT_SIZE)):
#     y_[i][j] = int(n)

###############################################################################
#      ONE HOT REPRESENTATION OF Y
Y = lfw_people.target
yOneHot = np.zeros((INPUT_SIZE, OUTPUT_SIZE),  dtype=np.int)
print(len(Y))
for index,target in enumerate(Y):  
  yOneHot[index, target-1] = 1
  #print(str(index) + " " + str(target))
##############################################################################

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 220*220])
y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])

W = tf.Variable(tf.zeros([220*220,OUTPUT_SIZE]))
b = tf.Variable(tf.zeros([OUTPUT_SIZE]))





def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv1(x, W):
  return tf.nn.conv2d(x, W, strides=[2, 2, 2, 2], padding='SAME')

def conv2(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_1(x):
  return tf.nn.max_pool(x, ksize=[3, 3, 64, 64], strides=[2, 2, 2, 2],padding='SAME')

def max_pool_2(x):
  return tf.nn.max_pool(x, ksize=[3, 3, 192, 64], strides=[1, 1, 1, 1],padding='SAME')

W_conv1 = weight_variable([7, 7, 3, 64])
b_conv1 = bias_variable([64])


x_image = tf.reshape(x, [-1,220,220,3])


h_conv1 = tf.nn.relu(conv1(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_1(h_conv1)



W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2(h_conv2)



W_fc1 = weight_variable([28* 28* 192, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 55* 55* 192])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


W_fc2 = weight_variable([1024, OUTPUT_SIZE])
b_fc2 = bias_variable([OUTPUT_SIZE])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2



cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))


train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


correct_prediction = tf.equal(tf.argmax(y_conv), tf.argmax(y_))


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess.run(tf.global_variables_initializer())

seventyfive = int(round(LENGTH*0.75))

for i in range(10):
  
  batch_xs = np.squeeze(np.array([lfw_people.data[0:seventyfive,:]]))
  batch_ys = np.squeeze(np.array([yOneHot[0:seventyfive,:]]))
  print(batch_ys.shape)
  print(batch_xs.shape)
  if i > 0:
    train_accuracy = accuracy.eval(feed_dict={
    	#TODO
        x:batch_xs, y_: batch_ys, keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: np.squeeze(np.array([lfw_people.data[seventyfive+1: LENGTH-1]])), y_: np.squeeze(np.array([yOneHot[seventyfive+1:LENGTH-1]])), keep_prob: 1.0}))
