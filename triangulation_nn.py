import tensorflow as tf
from tensorflow.python.framework.errors_impl import NotFoundError

from gen_dataset import data_gen_batch

x = tf.placeholder(tf.float32, [None, 4])



def model(x, hidden_units):
    def add_layer(imodel, layer_shape, st_dev=0.8):
        m = int(imodel.get_shape()[1])
        layer = tf.Variable(tf.truncated_normal([m, layer_shape], stddev=st_dev, mean=1))
        return tf.nn.relu(tf.matmul(imodel, layer))

    for hidden in hidden_units:
        m = add_layer(x, hidden)
    b = tf.Variable(tf.zeros([hidden_units[-1]]))
    return m + b
OUTS = 1
y = model(x, [4, 64, 32, OUTS])

y_ = tf.placeholder(tf.float32, [None, OUTS])

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.square(y_ * tf.log(y)), reduction_indices=[1]))
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-2,1.0))))

# cross_entropy = tf.reduce_mean(
#       tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
cross_entropy = tf.reduce_mean(tf.abs(y_ - y))
# cross_entropy = -tf.reduce_mean(tf.abs(y_ - y))


correct_prediction = tf.equal(y, y_)
accuracy = tf.reduce_mean(100*tf.abs(y -y_))
tf.summary.scalar("accuracy", accuracy)
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
try:
    saver.restore(sess, "weights/model.ckpt")
except NotFoundError:
    print("no model found, creating new model")

for _ in range(0, 1000):

    batch_x, batch_y = data_gen_batch(100)

    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

    print("epoch ", _, "accuracy", sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y}))

    #print("accuracy ", sess.run(cross_entropy, feed_dict={x: batch_x, y_: batch_y}))
    # print(sess.run(W0))

save_path = saver.save(sess, "weights/model.ckpt")

val_x, val_y = data_gen_batch(1000)
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(sess.run(accuracy, feed_dict={x: val_x, y_: val_y}))
print(val_x[0], val_y[0])
print(sess.run(y, feed_dict={x: [val_x[0]]}))





