import tensorflow as tf
import numpy as np
from draw_func import project_dot2d
from random import randrange
# def trinagulation_nn(x1, x2, c1, c2, viewer):
#     xg = tf.placeholder(tf.int32, [None, 12])
#     yg = tf.placeholder(tf.int32, [None, 3])
#
#     W = tf.Variable(tf.zeros([30, 3]))
#     b = tf.Variable(tf.zeros([3]))
#     y = tf.matmul(x, W) + b
#
#
# def train_network(xi, yi):
#     cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


def data_gen(examples):
    cam1 = [randrange(1, 100), randrange(1, 100), randrange(1, 9)]
    cam2 = [randrange(1, 100), randrange(1, 100), randrange(1, 9)]
    # cam = np.array((122, 122, 1))
    viewer = [0, 0, -2]
    theta = (0, 0, 0)

    for x in range(0, examples):
        d1 = [randrange(1, 100), randrange(1, 100), randrange(11, 100)]
        proj1 = project_dot2d(np.array(d1), np.array(viewer), np.array(cam1), np.array(theta))
        proj2 = project_dot2d(np.array(d1), np.array(viewer), np.array(cam2), np.array(theta))

        res = list(proj1) + list(proj2) + cam1 + cam2 + viewer[2:]
        yield res, d1

def data_gen_batch(batch_size):
    inp = []
    out = []
    for x, y in data_gen(batch_size):
        inp.append(x)
        out.append(y)

    return np.array(inp).astype(np.float32) / 1000., np.array(out).astype(np.float32) / 1000.



x = tf.placeholder(tf.float32, [None, 11])

W0 = tf.Variable(tf.zeros([11, 3]))
#W1 = tf.Variable(tf.truncated_normal([50, 3], stddev=0.1))
#W0 = tf.Variable(tf.truncated_normal([11, 3], stddev=0.1))
b = tf.Variable(tf.zeros([3]))

y = tf.matmul(x, W0) + b


y_ = tf.placeholder(tf.float32, [None, 3])

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(y, y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("accuracy", accuracy)

for _ in range(0, 10000):

    batch_x, batch_y = data_gen_batch(100)

    print("epoch ", _, cross_entropy)

    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})



val_x, val_y = data_gen_batch(1)
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(sess.run(accuracy, feed_dict={x: val_x, y_: val_y}))
print(val_x, val_y)
print(sess.run(y, feed_dict={x: val_x }))





