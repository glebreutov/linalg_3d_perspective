import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors_impl import NotFoundError

from dataset_tools import gen_batch
from scene import translation, intrinsic
from train_util import model

intr = intrinsic(focal=200, aspect_ratio=300, skew=0)

cam = np.array((400, 500, 2))
theta = np.array((0.05, 0, 0))
trans1 = translation(cam, theta)

cam = np.array((600, 500, 2))
theta = np.array((-0.05, 0, 0))
trans2 = translation(cam, theta)

MODEL_NAME = "stereo_angle/trin_stereo_angle.ckpt"

OUTS = 3
INS = 4
STDEV = 500.

x = tf.placeholder(tf.float32, [None, INS])
y = model(x, [INS, 32, 32, 32, OUTS], tf.nn.tanh)
y_ = tf.placeholder(tf.float32, [None, OUTS])

sess = tf.Session()

saver = tf.train.Saver()

# cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.abs(y_[:, 2] - y[:, 2])))
cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.abs(y_ - y), reduction_indices=[1]))
# cross_entropy = tf.losses.mean_pairwise_squared_error(y, y_)

train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
sess.run(tf.global_variables_initializer())

try:
    saver.restore(sess, MODEL_NAME)
    print(MODEL_NAME, "loaded")
except NotFoundError:
    print("no model found, creating new model")


for i in range(0, 10000):
    batch_x, batch_y = gen_batch(intr, trans1, trans2, STDEV, 50)
    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
    if i % 100 == 0:
        accuracy = sess.run(cross_entropy, feed_dict={x: batch_x, y_: batch_y})
        print("batch ", i, "accuracy", accuracy)

save_path = saver.save(sess, MODEL_NAME)
print(save_path)

test_x, test_y = gen_batch(intr, trans1, trans2, STDEV, 100)
# print("test x", test_x)
# print("test y", test_y)
sess.run(y, feed_dict={x: test_x})
accuracy = sess.run(cross_entropy, feed_dict={x: test_x, y_: test_y})
print("test batch accuracy", accuracy)

test_x, test_y = gen_batch(intr, trans1, trans2, STDEV, 10)
y = sess.run(y, feed_dict={x: test_x})
for i in range(0, len(test_x)):
    print(test_y[i], np.round(y[i], 3), sum(abs(y[i] - test_y[i])))

