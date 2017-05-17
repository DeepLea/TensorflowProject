"""
    Skrev om MNIST-exemplet och importerar lfw från sklearn
    Det funkar inte riktigt än, har markerat med TODO
"""
###############################################################################
#      IMPORTS
import tensorflow.contrib.learn as skflow
from sklearn.datasets import fetch_lfw_people
import numpy as np
import tensorflow as tf
lfw_people = fetch_lfw_people(min_faces_per_person = 70, resize = 0.4)  # Det här tar tid, upp till 2 min

INPUT_SIZE = lfw_people.data.shape[1]
OUTPUT_SIZE = 6 # There are > 6000 unique persons: 2^13 > 6000


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

for index, target in enumerate(Y):
  yOneHot[index, target - 1] = 1


###############################################################################
#      Setting the placeholders for the variables and operators
x = tf.placeholder(tf.float32, [None, INPUT_SIZE])
W = tf.Variable(tf.zeros([INPUT_SIZE, OUTPUT_SIZE]))
b = tf.Variable(tf.zeros([OUTPUT_SIZE]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run() # Init the variable_scope


###############################################################################
#      Setting the placeholders for the variables and operators

# TODO: fix this batch-learning loop
for i in range(1000):
  # batch_xs, batch_ys = mnist.train.next_batch(100) # Såhär hämtade de från exemplet
  batch_xs = np.array([lfw_people.data[i, :]])

  batch_ys = np.array([yOneHot[i, :]])

  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

print("TRANING: DONE!")
print()
print("---= STARTING TEST =---")
###############################################################################
#      Printing the results
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

avr = []

for i in range(1001, 1200):
  xTest = np.array([lfw_people.data[i, :]])
  yTest = np.array([yOneHot[i, :]])

  result = sess.run(accuracy, feed_dict={x: xTest, y_: yTest})

  # print(result)

  avr.append(result)

print("SLUTRESULTAT:")
print(np.mean(avr))

# print(sess.run(accuracy, feed_dict={x: xTest, y_: yTest}))
