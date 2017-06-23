import tensorflow as tf

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
from gen_dataset import data_gen_batch

x = tf.placeholder(tf.float32, [None, 4])

# W0 = tf.Variable(tf.truncated_normal([4, 4], stddev=1))
# W1 = tf.Variable(tf.truncated_normal([4, 3], stddev=1))
# #W0 = tf.Variable(tf.truncated_normal([11, 3], stddev=0.1))
#
#
# y = tf.matmul(tf.matmul(x, W0), W1) + b

# [n X m] * [m X p] =  [n X p]


def model(x, hidden_units):
    def add_layer(imodel, layer_shape, st_dev=0.01):
        m = int(imodel.get_shape()[1])
        layer = tf.Variable(tf.truncated_normal([m, layer_shape], stddev=st_dev))
        return tf.nn.relu(tf.matmul(imodel, layer))

    for hidden in hidden_units:
        m = add_layer(x, hidden)
    b = tf.Variable(tf.zeros([hidden_units[-1]]))
    return m + b

y = model(x, [4, 1024, 64, 32, 3])

y_ = tf.placeholder(tf.float32, [None, 3])

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.square(y_ * tf.log(y)), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0))))

# cross_entropy = tf.reduce_sum(y_ - y) / 100
# cross_entropy = -tf.reduce_mean(tf.abs(y_ - y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_prediction = tf.equal(y, y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("accuracy", accuracy)

for _ in range(0, 1000):

    batch_x, batch_y = data_gen_batch(100)

    print("epoch ", _)

    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
    print("accuracy ", sess.run(cross_entropy, feed_dict={x: batch_x, y_: batch_y}))
    # print(sess.run(W0))


val_x, val_y = data_gen_batch(1000)
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(sess.run(accuracy, feed_dict={x: val_x, y_: val_y}))
print(val_x[0], val_y[0])
print(sess.run(y, feed_dict={x: [val_x[0]] }))





