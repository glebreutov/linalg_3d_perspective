import tensorflow as tf

from gen_dataset import data_gen_batch
from train_util import model


def triangulate():
    OUTS = 3
    INS = 10
    MODEL_NAME = "weights/model_back_to_roots_z.ckpt"

    x = tf.placeholder(tf.float32, [None, INS])
    y = model(x, [INS, 32, 32, OUTS])

    sess = tf.Session()

    saver = tf.train.Saver()
    saver.restore(sess, MODEL_NAME)
    while True:
        proj = yield
        yield sess.run(y, feed_dict={x: [proj]})


batch_x, batch_y = data_gen_batch(20)
print(batch_y[0])
fx = triangulate()
fx.send(None)
print(fx.send(batch_x[0]))
print(fx.send(batch_x[1]))
print(fx.send(batch_x[2]))
