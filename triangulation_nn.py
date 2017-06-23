import tensorflow as tf
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.summary.writer.writer import FileWriter

from gen_dataset import data_gen_batch
from train_util import model

OUTS = 3
INS = 8
MODEL_NAME = "weights/model_quad_linear.ckpt"

x = tf.placeholder(tf.float32, [None, INS])

y = model(x, [INS, 32, 32, OUTS])

y_ = tf.placeholder(tf.float32, [None, OUTS])

sess = tf.Session()


saver = tf.train.Saver()

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.square(y_ * tf.log(y)), reduction_indices=[1]))
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-2,1.0))))

# cross_entropy = tf.reduce_mean(
#       tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
cross_entropy = tf.reduce_mean(tf.abs(y_ - y))
# cross_entropy = -tf.reduce_mean(tf.abs(y_ - y))


correct_prediction = tf.equal(y, y_)
accuracy = tf.reduce_mean(100*tf.abs(y -y_))
summary_accuracy = tf.summary.scalar("accuracy", accuracy)
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
fw = FileWriter("log", sess.graph)


sess.run(tf.global_variables_initializer())
try:
    saver.restore(sess, MODEL_NAME)
except NotFoundError:
    print("no model found, creating new model")
merged = tf.summary.merge_all()

for _ in range(0, 100):

    batch_x, batch_y = data_gen_batch(100)

    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

    a, b = sess.run([merged, accuracy], feed_dict={x: batch_x, y_: batch_y})
    fw.add_summary(a, _)
    print("epoch ", _, "accuracy", b)

    #print("accuracy ", sess.run(cross_entropy, feed_dict={x: batch_x, y_: batch_y}))
    # print(sess.run(W0))

save_path = saver.save(sess, MODEL_NAME)

val_x, val_y = data_gen_batch(1000)
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(sess.run(accuracy, feed_dict={x: val_x, y_: val_y}))
print(val_x[0], val_y[0])
print(sess.run(y, feed_dict={x: [val_x[0]]}))





