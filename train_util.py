import tensorflow as tf


def model(x, hidden_units, activation):
    def add_layer(imodel, layer_shape, st_dev=0.8):
        m = int(imodel.get_shape()[1])
        layer = tf.Variable(tf.truncated_normal([m, layer_shape], stddev=st_dev, mean=0))
        b = tf.Variable(tf.zeros([layer_shape]))
        return activation(tf.matmul(imodel, layer) + b)

    m = x
    for hidden in hidden_units:
        m = add_layer(m, hidden)
    b = tf.Variable(tf.zeros([hidden_units[-1]]))
    return m + b
