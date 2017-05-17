
# IMPORTERAR SAKER

from __future__ import print_function

from time import time

import tensorflow as tf
import numpy as np
import random
import sys

#TIME
t0 = time()

#LASER IN DATA

data = np.load('data.npy')
target = np.load('target.npy')
imagesOld = np.load('images.npy')
names = np.load('names.npy')
DESCR = np.load('DESCR.npy')

#OMFORMATERAR DATA
images = imagesOld.flatten().reshape(1288, 1850)

print(DESCR)
no_names = len(names)

# PLACEHOLDERS

#x = tf.placeholder(tf.float32, [1850,None])
x = tf.placeholder(tf.float32, [1850])
y_ = tf.placeholder(tf.float32, [no_names])


#ONE HOT REPRESENTATION FROM VALUES
a = list()

for i in range(0,1287):
	temp = [0,0,0,0,0,0,0]
	temp[target[i]] = 1
	a.append(list(temp))
	#print(temp)
target = a

# W & b placeholders?

W = tf.Variable(tf.zeros([1850, no_names]))
b = tf.Variable(tf.zeros([no_names]))

# EXPANDERAR X MED EN DIMENTION FOR ATT DET SKA FUNKA
y = tf.nn.softmax(tf.matmul(tf.expand_dims(x,0), W) + b)
#y = tf.nn.softmax(tf.matmul(x, W) + b)

#CROSS ENTROPY
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

#FOR LOOP PLOCKAR UT VARJE BILD (GAR INTE ATT TA FLERA AV NGN ANLEDNING)
for _ in range(1): # antal loops?
	for i in range(1287):
  		batch_xs = images[i]
  		batch_ys = target[i]
  		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y), tf.argmax(y_))

## ACCURACY BLIR 0
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
totAcc = float()
for i in range(1287):
	acc = sess.run(accuracy, feed_dict={x: images[i], y_: target[i]})
	#print(i)
	totAcc += np.asscalar(acc)
print("Accuracy after 1 loop is: " + str(totAcc/1288))


