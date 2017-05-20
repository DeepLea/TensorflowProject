##############################################################################
# TAGS
__year__ = 2017
__auth__ = "Per Jaakonantti, Marcus Wallberg & Johanna Gustafsson"
##############################################################################
# IMPORTING

import tensorflow as tf
import numpy as np

##############################################################################
# LOADING DATA

data = np.load("data.npy")
target = np.load("target.npy")
target_names = np.load("target_names.npy")

##############################################################################
# SETTING SIZES

INPUT_SIZE = data.shape[1]
OUTPUT_SIZE = len(target_names)-1
LENGTH = len(data)

##############################################################################
# CONVERTING TO ONE HOT REPRESENTATION OF Y

Y = target
yOneHot = np.zeros((INPUT_SIZE, OUTPUT_SIZE),  dtype=np.int)

for index,target in enumerate(Y):  
  yOneHot[index, target-1] = 1

##############################################################################
# INITIALISING SESSION AND x, y_, W AND b

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 220*220])
y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])

W = tf.Variable(tf.zeros([220*220,OUTPUT_SIZE]))
b = tf.Variable(tf.zeros([OUTPUT_SIZE]))

##############################################################################
# DEFINING FUNCTIONS

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv(x, W, stride):
  if stride == 1:
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
  elif stride == 2:
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME')

##############################################################################
# DEFINING ALL LAYERS

# FIRST CONV LAYER AND FIRST POOL
W_conv1 = weight_variable([7, 7, 1, 64])
b_conv1 = bias_variable([64])

x_image = tf.reshape(x, [-1,220,220,1])

h_conv1 = tf.nn.relu(conv(x_image, W_conv1,2) + b_conv1)
h_pool1 = max_pool(h_conv1)

#SECOND CONV + 2a AND SECOND POOL
W_conv2a = weight_variable([1, 1, 64, 64])
b_conv2a = bias_variable([64])
W_conv2 = weight_variable([3, 3, 64, 192])
b_conv2 = bias_variable([192])

h_conv2a = tf.nn.relu(conv(h_pool1, W_conv2a,1) + b_conv2a)
h_conv2 = tf.nn.relu(conv(h_conv2a, W_conv2,1) + b_conv2)
h_pool2 = max_pool(h_conv2)

#THIRD CONV + 3a AND THIRD POOL
W_conv3a = weight_variable([1, 1, 192, 192])
b_conv3a = bias_variable([192])
W_conv3 = weight_variable([3, 3, 192, 384])
b_conv3 = bias_variable([384])

h_conv3a = tf.nn.relu(conv(h_pool2, W_conv3a,1) + b_conv3a)
h_conv3 = tf.nn.relu(conv(h_conv3a, W_conv3,1) + b_conv3)
h_pool3 = max_pool(h_conv3)

#FOURTH CONV + 4a
W_conv4a = weight_variable([1, 1, 384, 384])
b_conv4a = bias_variable([384])
W_conv4 = weight_variable([3, 3, 384, 256])
b_conv4 = bias_variable([256])

h_conv4a = tf.nn.relu(conv(h_pool3, W_conv4a,1) + b_conv4a)
h_conv4 = tf.nn.relu(conv(h_conv4a, W_conv4,1) + b_conv4)

#FIFTH CONV + 5a
W_conv5a = weight_variable([1, 1, 256, 256])
b_conv5a = bias_variable([256])
W_conv5 = weight_variable([3, 3, 256, 256])
b_conv5 = bias_variable([256])

h_conv5a = tf.nn.relu(conv(h_conv4, W_conv5a,1) + b_conv5a)
h_conv5 = tf.nn.relu(conv(h_conv5a, W_conv5,1) + b_conv5)

#SIXTH CONV + 6a AND FOURTH POOL
W_conv6a = weight_variable([1, 1, 256, 256])
b_conv6a = bias_variable([256])
W_conv6 = weight_variable([3, 3, 256, 256])
b_conv6 = bias_variable([256])

h_conv6a = tf.nn.relu(conv(h_conv5, W_conv6a,1) + b_conv6a)
h_conv6 = tf.nn.relu(conv(h_conv6a, W_conv6,1) + b_conv6)
h_pool4 = max_pool(h_conv6)

# FULLY CONNECTED LAYER 1 AND 2

W_fc1 = weight_variable([7* 7* 256, 4096])
b_fc1 = bias_variable([4096])

h_pool4_flat = tf.reshape(h_pool4, [-1, 7* 7* 256])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([4096, OUTPUT_SIZE])
b_fc2 = bias_variable([OUTPUT_SIZE])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

##############################################################################
# DEFINING ACCURACY & CROSS ENTROPY

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

##############################################################################
# LOOP FOR TRAINING

seventyfive = int(round(LENGTH*0.75))
batchsize = 64
epochs = 10
trainacc = [0] * (epochs)
traincost = [0] * (epochs)
testacc = [0] * (epochs)
testcost = [0] * (epochs)

for j in range(epochs):
  for i in range(seventyfive/batchsize + 1): 
    if i == seventyfive/batchsize:
      batch_xs = np.squeeze(np.array([data[i*batchsize:seventyfive,:]]))
      batch_ys = np.squeeze(np.array([yOneHot[i*batchsize:seventyfive,:]]))
    else:
      batch_xs = np.squeeze(np.array([data[i*batchsize:i*batchsize+batchsize-1,:]]))
      batch_ys = np.squeeze(np.array([yOneHot[i*batchsize:i*batchsize+batchsize-1,:]]))
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
  train_accuracy = accuracy.eval(feed_dict={x:np.squeeze(np.array([data[0:seventyfive,:]])), y_: np.squeeze(np.array([yOneHot[0:seventyfive,:]])), keep_prob: 1.0})
  trainacc[j] = train_accuracy
  testacc[j] = accuracy.eval(feed_dict={x: np.squeeze(np.array([data[seventyfive+1: LENGTH-1]])), y_: np.squeeze(np.array([yOneHot[seventyfive+1:LENGTH-1]])), keep_prob: 1.0})
  traincost[j] = cross_entropy.eval(feed_dict={x: np.squeeze(np.array([data[0: seventyfive]])), y_: np.squeeze(np.array([yOneHot[0:seventyfive]])), keep_prob: 1.0})
  testcost[j] = cross_entropy.eval(feed_dict={x: np.squeeze(np.array([data[seventyfive+1: LENGTH-1]])), y_: np.squeeze(np.array([yOneHot[seventyfive+1:LENGTH-1]])), keep_prob: 1.0})
  print("step %d, training accuracy %g, test accuracy %g, test cost %g"%(j, train_accuracy,testacc[j], testcost[j]))
  
##############################################################################
# GETTING FINAL ACCURACY

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: np.squeeze(np.array([data[seventyfive+1: LENGTH-1]])), y_: np.squeeze(np.array([yOneHot[seventyfive+1:LENGTH-1]])), keep_prob: 1.0}))

##############################################################################
# CALCULATION INDIVIDUAL ACCURACY

personNum = [0] * (OUTPUT_SIZE+1)
personRig = [0] * (OUTPUT_SIZE+1)

for i in range(seventyfive+1, LENGTH-1):
  accTemp = accuracy.eval(feed_dict={x: np.array([data[i]]), y_: np.array([yOneHot[i]]), keep_prob: 1.0})
  person = Y[i]
  personNum[person-1] += 1
  personRig[person-1] += accTemp

for i in range(0,OUTPUT_SIZE):
  print(target_names[i] + ": " + str(100*float(personRig[i])/float(personNum[i])) + " % Accuracy")

##############################################################################
# SAVING VARIABLES

np.save("trainCost.npy", traincost)
np.save("testCost.npy", testcost)
np.save("trainAcc.npy", trainacc)
np.save("testAcc.npy", testacc)
np.save("personNum.npy",personNum)
np.save("personRig.npy",personRig)

