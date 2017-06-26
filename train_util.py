import tensorflow as tf


def model(x, hidden_units):
    def add_layer(imodel, layer_shape, st_dev=0.8):
        m = int(imodel.get_shape()[1])
        layer = tf.Variable(tf.truncated_normal([m, layer_shape], stddev=st_dev, mean=0))
        return tf.nn.relu(tf.matmul(imodel, layer))

    for hidden in hidden_units:
        m = add_layer(x, hidden)
    b = tf.Variable(tf.zeros([hidden_units[-1]]))
    return m + b